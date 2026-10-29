#include "dogperc/npz.hpp"

#include <zlib.h>

#include <fstream>
#include <sstream>

namespace dogperc {

namespace {

uint16_t u16(const uint8_t* p) { return static_cast<uint16_t>(p[0] | (p[1] << 8)); }
uint32_t u32(const uint8_t* p) {
  return static_cast<uint32_t>(p[0]) | (static_cast<uint32_t>(p[1]) << 8) | (static_cast<uint32_t>(p[2]) << 16) |
         (static_cast<uint32_t>(p[3]) << 24);
}
uint64_t u64(const uint8_t* p) { return static_cast<uint64_t>(u32(p)) | (static_cast<uint64_t>(u32(p + 4)) << 32); }

void put16(std::vector<uint8_t>& b, uint16_t v) {
  b.push_back(v & 0xff);
  b.push_back(v >> 8);
}
void put32(std::vector<uint8_t>& b, uint32_t v) {
  for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xff);
}

// value of 'key' in the .npy header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
std::string header_value(const std::string& h, const std::string& key) {
  const size_t k = h.find("'" + key + "'");
  if (k == std::string::npos) throw Error("npy header without " + key + ": " + h);
  size_t p = h.find(':', k) + 1;
  while (p < h.size() && h[p] == ' ') ++p;
  if (h[p] == '\'') return h.substr(p + 1, h.find('\'', p + 1) - p - 1);
  if (h[p] == '(') return h.substr(p + 1, h.find(')', p) - p - 1);
  size_t e = p;
  while (e < h.size() && h[e] != ',' && h[e] != '}') ++e;
  return h.substr(p, e - p);
}

std::vector<uint8_t> inflate_raw(const uint8_t* src, size_t n, size_t out_size) {
  std::vector<uint8_t> out(out_size);
  z_stream zs{};
  if (inflateInit2(&zs, -MAX_WBITS) != Z_OK) throw Error("zlib inflateInit2 failed");
  zs.next_in = const_cast<Bytef*>(src);
  zs.avail_in = static_cast<uInt>(n);
  zs.next_out = out.data();
  zs.avail_out = static_cast<uInt>(out.size());
  const int r = inflate(&zs, Z_FINISH);
  inflateEnd(&zs);
  if (r != Z_STREAM_END || zs.total_out != out_size) throw Error("npz: corrupt deflate stream");
  return out;
}

}  // namespace

int64_t NpyArray::size() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

int NpyArray::itemsize() const { return std::stoi(descr.substr(2)); }

char NpyArray::kind() const { return descr.size() > 1 ? descr[1] : '?'; }

std::vector<uint8_t> read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw Error("cannot open " + path);
  return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

NpyArray parse_npy(const uint8_t* d, size_t n) {
  if (n < 10 || std::memcmp(d, "\x93NUMPY", 6) != 0) throw Error("not an .npy array");
  const int major = d[6];
  size_t hlen, off;
  if (major == 1) {
    hlen = u16(d + 8);
    off = 10;
  } else {
    hlen = u32(d + 8);
    off = 12;
  }
  const std::string h(reinterpret_cast<const char*>(d + off), hlen);
  NpyArray a;
  a.descr = header_value(h, "descr");
  if (a.descr.size() < 3 || (a.descr[0] != '<' && a.descr[0] != '|'))
    throw Error("npy: only little-endian numeric arrays are supported, got " + a.descr);
  a.fortran_order = header_value(h, "fortran_order").find("True") != std::string::npos;
  std::stringstream ss(header_value(h, "shape"));
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (tok.find_first_not_of(' ') == std::string::npos) continue;
    a.shape.push_back(std::stoll(tok));
  }
  const size_t nbytes = static_cast<size_t>(a.size()) * a.itemsize();
  if (off + hlen + nbytes > n) throw Error("npy: truncated data");
  a.bytes.assign(d + off + hlen, d + off + hlen + nbytes);
  if (a.fortran_order && a.shape.size() > 1) throw Error("npy: Fortran-ordered arrays are not supported");
  return a;
}

NpyArray read_npy(const std::string& path) {
  const auto b = read_file(path);
  return parse_npy(b.data(), b.size());
}

Npz read_npz(const std::string& path) {
  const auto z = read_file(path);
  const size_t n = z.size();
  // end of central directory: the last "PK\5\6"
  size_t eocd = std::string::npos;
  if (n >= 22) {
    const size_t lo = n > 22 + 65535 ? n - 22 - 65535 : 0;    // the record ends with a comment of <= 64 KiB
    for (size_t i = n - 22 + 1; i-- > lo;) {
      if (u32(&z[i]) == 0x06054b50) {
        eocd = i;
        break;
      }
    }
  }
  if (eocd == std::string::npos) throw Error(path + ": not a zip / npz file");
  uint64_t count = u16(&z[eocd + 10]);
  uint64_t cd_off = u32(&z[eocd + 16]);
  if ((count == 0xffff || cd_off == 0xffffffff) && eocd >= 20 && u32(&z[eocd - 20]) == 0x07064b50) {
    const uint64_t z64 = u64(&z[eocd - 20 + 8]);          // ZIP64 end of central directory record
    count = u64(&z[z64 + 32]);
    cd_off = u64(&z[z64 + 48]);
  }
  Npz out;
  size_t p = cd_off;
  for (uint64_t e = 0; e < count; ++e) {
    if (p + 46 > n || u32(&z[p]) != 0x02014b50) throw Error(path + ": bad central directory");
    const uint16_t method = u16(&z[p + 10]);
    uint64_t csize = u32(&z[p + 20]), usize = u32(&z[p + 24]);
    const uint16_t name_len = u16(&z[p + 28]), extra_len = u16(&z[p + 30]), comment_len = u16(&z[p + 32]);
    uint64_t local = u32(&z[p + 42]);
    std::string name(reinterpret_cast<const char*>(&z[p + 46]), name_len);
    const uint8_t* ex = &z[p + 46 + name_len];
    for (size_t q = 0; q + 4 <= extra_len;) {                 // ZIP64 extra field: the sizes that are 0xffffffff
      const uint16_t id = u16(ex + q), len = u16(ex + q + 2);
      if (id == 0x0001) {
        size_t r = q + 4;
        if (usize == 0xffffffff) { usize = u64(ex + r); r += 8; }
        if (csize == 0xffffffff) { csize = u64(ex + r); r += 8; }
        if (local == 0xffffffff) { local = u64(ex + r); r += 8; }
      }
      q += 4 + len;
    }
    p += 46 + name_len + extra_len + comment_len;
    if (u32(&z[local]) != 0x04034b50) throw Error(path + ": bad local header");
    const size_t data = local + 30 + u16(&z[local + 26]) + u16(&z[local + 28]);
    if (data + csize > n) throw Error(path + ": truncated member " + name);
    std::vector<uint8_t> raw;
    if (method == 0) raw.assign(z.begin() + data, z.begin() + data + csize);
    else if (method == 8) raw = inflate_raw(&z[data], csize, usize);
    else throw Error(path + ": unsupported zip compression method");
    if (name.size() > 4 && name.compare(name.size() - 4, 4, ".npy") == 0) name.resize(name.size() - 4);
    out[name] = parse_npy(raw.data(), raw.size());
  }
  return out;
}

const NpyArray& member(const Npz& npz, const std::string& name) {
  auto it = npz.find(name);
  if (it == npz.end()) throw Error("npz has no member '" + name + "'");
  return it->second;
}

template <> const char* npy_descr<float>() { return "<f4"; }
template <> const char* npy_descr<double>() { return "<f8"; }
template <> const char* npy_descr<int64_t>() { return "<i8"; }
template <> const char* npy_descr<int32_t>() { return "<i4"; }
template <> const char* npy_descr<uint8_t>() { return "|u1"; }
template <> const char* npy_descr<bool>() { return "|b1"; }

void NpzWriter::add_raw(const std::string& name, const std::string& descr, const std::vector<int64_t>& shape,
                        const void* data, size_t nbytes) {
  std::string s;                                             // (n,) / (n, m) / () as Python writes them
  for (size_t i = 0; i < shape.size(); ++i) {
    s += std::to_string(shape[i]);
    if (shape.size() == 1 || i + 1 < shape.size()) s += shape.size() == 1 ? "," : ", ";
  }
  std::string h = "{'descr': '" + descr + "', 'fortran_order': False, 'shape': (" + s + "), }";
  const size_t total = 10 + h.size() + 1;
  h += std::string((64 - total % 64) % 64, ' ') + "\n";      // header padded to a multiple of 64
  std::vector<uint8_t> b = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
  put16(b, static_cast<uint16_t>(h.size()));
  b.insert(b.end(), h.begin(), h.end());
  const uint8_t* d = static_cast<const uint8_t*>(data);
  b.insert(b.end(), d, d + nbytes);
  members_.emplace_back(name + ".npy", std::move(b));
}

void NpzWriter::write(const std::string& path) const {
  std::vector<uint8_t> out, cd;
  for (const auto& [name, data] : members_) {
    const uint32_t crc = crc32(0L, data.data(), static_cast<uInt>(data.size()));
    const uint32_t off = static_cast<uint32_t>(out.size());
    put32(out, 0x04034b50);
    put16(out, 20);
    put16(out, 0);
    put16(out, 0);       // stored
    put16(out, 0);
    put16(out, 0x21);    // 1980-01-01
    put32(out, crc);
    put32(out, static_cast<uint32_t>(data.size()));
    put32(out, static_cast<uint32_t>(data.size()));
    put16(out, static_cast<uint16_t>(name.size()));
    put16(out, 0);
    out.insert(out.end(), name.begin(), name.end());
    out.insert(out.end(), data.begin(), data.end());
    put32(cd, 0x02014b50);
    put16(cd, 20);
    put16(cd, 20);
    put16(cd, 0);
    put16(cd, 0);
    put16(cd, 0);
    put16(cd, 0x21);
    put32(cd, crc);
    put32(cd, static_cast<uint32_t>(data.size()));
    put32(cd, static_cast<uint32_t>(data.size()));
    put16(cd, static_cast<uint16_t>(name.size()));
    put16(cd, 0);
    put16(cd, 0);
    put16(cd, 0);
    put16(cd, 0);
    put32(cd, 0);
    put32(cd, off);
    cd.insert(cd.end(), name.begin(), name.end());
  }
  const uint32_t cd_off = static_cast<uint32_t>(out.size());
  out.insert(out.end(), cd.begin(), cd.end());
  put32(out, 0x06054b50);
  put16(out, 0);
  put16(out, 0);
  put16(out, static_cast<uint16_t>(members_.size()));
  put16(out, static_cast<uint16_t>(members_.size()));
  put32(out, static_cast<uint32_t>(cd.size()));
  put32(out, cd_off);
  put16(out, 0);
  std::ofstream f(path, std::ios::binary);
  if (!f) throw Error("cannot write " + path);
  f.write(reinterpret_cast<const char*>(out.data()), static_cast<std::streamsize>(out.size()));
}

}  // namespace dogperc
