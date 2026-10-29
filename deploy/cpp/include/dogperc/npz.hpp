// NumPy .npy / .npz files: the sequences (scans/*.npz, imu.npy, odom.npy), the golden data of the parity
// tests and the replay output. Little-endian numeric dtypes only; .npz members stored or deflated (what
// np.savez / np.savez_compressed write, ZIP64 included).
#pragma once

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "dogperc/common.hpp"

namespace dogperc {

struct NpyArray {
  std::string descr;                // NumPy dtype string: "<f4", "<f8", "<i8", "|b1", ...
  std::vector<int64_t> shape;
  bool fortran_order = false;
  std::vector<uint8_t> bytes;

  int64_t size() const;             // number of elements
  int itemsize() const;
  char kind() const;                // 'f', 'i', 'u', 'b'
  int64_t dim(int i) const { return shape.at(i); }

  // element i converted to T (any numeric dtype)
  template <class T>
  T at(int64_t i) const;
  // every element converted to T
  template <class T>
  std::vector<T> to() const {
    std::vector<T> out(size());
    for (int64_t i = 0; i < size(); ++i) out[i] = at<T>(i);
    return out;
  }
  double scalar() const { return at<double>(0); }
};

NpyArray parse_npy(const uint8_t* data, size_t n);
NpyArray read_npy(const std::string& path);
using Npz = std::map<std::string, NpyArray>;
Npz read_npz(const std::string& path);
// member of an .npz, with a clear error when missing
const NpyArray& member(const Npz& npz, const std::string& name);

template <class T>
const char* npy_descr();   // "<f4" for float, "<f8" for double, "<i8" for int64_t, "<i4" for int32_t, "|u1", "|b1"

// .npz writer (stored, no compression): np.load reads it
class NpzWriter {
 public:
  template <class T>
  void add(const std::string& name, const std::vector<T>& values, std::vector<int64_t> shape = {}) {
    if (shape.empty()) shape = {static_cast<int64_t>(values.size())};
    add_raw(name, npy_descr<T>(), shape, values.data(), values.size() * sizeof(T));
  }
  template <class T>
  void add_scalar(const std::string& name, T value) {
    add_raw(name, npy_descr<T>(), {}, &value, sizeof(T));
  }
  void add_raw(const std::string& name, const std::string& descr, const std::vector<int64_t>& shape,
               const void* data, size_t nbytes);
  void write(const std::string& path) const;

 private:
  std::vector<std::pair<std::string, std::vector<uint8_t>>> members_;
};

std::vector<uint8_t> read_file(const std::string& path);

template <class T>
T NpyArray::at(int64_t i) const {
  const uint8_t* p = bytes.data() + i * itemsize();
  const char k = kind();
  const int s = itemsize();
  if (k == 'f') {
    if (s == 4) { float v; std::memcpy(&v, p, 4); return static_cast<T>(v); }
    if (s == 8) { double v; std::memcpy(&v, p, 8); return static_cast<T>(v); }
  } else if (k == 'i') {
    if (s == 1) { int8_t v; std::memcpy(&v, p, 1); return static_cast<T>(v); }
    if (s == 2) { int16_t v; std::memcpy(&v, p, 2); return static_cast<T>(v); }
    if (s == 4) { int32_t v; std::memcpy(&v, p, 4); return static_cast<T>(v); }
    if (s == 8) { int64_t v; std::memcpy(&v, p, 8); return static_cast<T>(v); }
  } else if (k == 'u' || k == 'b') {
    if (s == 1) { uint8_t v; std::memcpy(&v, p, 1); return static_cast<T>(v); }
    if (s == 2) { uint16_t v; std::memcpy(&v, p, 2); return static_cast<T>(v); }
    if (s == 4) { uint32_t v; std::memcpy(&v, p, 4); return static_cast<T>(v); }
    if (s == 8) { uint64_t v; std::memcpy(&v, p, 8); return static_cast<T>(v); }
  }
  throw Error("npy: unsupported dtype " + descr);
}

}  // namespace dogperc
