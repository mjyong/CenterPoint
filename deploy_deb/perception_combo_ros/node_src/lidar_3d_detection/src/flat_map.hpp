// Open-addressing hash set / map of non-negative int64 keys (linear probing, fixed capacity).
// The per-point hashing of dedup_returns / neighbor_counts spent most of its time allocating
// std::unordered_map nodes; these are flat arrays.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace lidar3d {

namespace detail {
constexpr int64_t kEmpty = -1;
inline size_t capacity_for(size_t n) {      // load factor <= 1/2
  size_t cap = 16;
  while (cap < 2 * n) cap <<= 1;
  return cap;
}
inline size_t hash(int64_t key) {           // murmur3 finalizer
  uint64_t k = static_cast<uint64_t>(key);
  k ^= k >> 33;
  k *= 0xff51afd7ed558ccdULL;
  k ^= k >> 33;
  k *= 0xc4ceb9fe1a85ec53ULL;
  k ^= k >> 33;
  return static_cast<size_t>(k);
}
}  // namespace detail

// keys only; reset() keeps the memory, so a set reused per sweep does not fault fresh pages every time
class FlatSet {
 public:
  void reset(size_t n) {
    const size_t cap = detail::capacity_for(n);
    if (keys_.size() == cap) {
      std::fill(keys_.begin(), keys_.end(), detail::kEmpty);
    } else {
      keys_.assign(cap, detail::kEmpty);
    }
    mask_ = cap - 1;
  }
  bool insert(int64_t key) {                // true when new
    size_t i = detail::hash(key) & mask_;
    while (keys_[i] != detail::kEmpty) {
      if (keys_[i] == key) return false;
      i = (i + 1) & mask_;
    }
    keys_[i] = key;
    return true;
  }

 private:
  std::vector<int64_t> keys_;
  size_t mask_ = 0;
};

class FlatMap {
 public:
  // room for `n` keys at load factor <= 1/2
  explicit FlatMap(size_t n) {
    const size_t cap = detail::capacity_for(n);
    keys_.assign(cap, detail::kEmpty);
    vals_.assign(cap, 0);
    mask_ = cap - 1;
  }
  // the value of `key`, inserted as 0 when new (`inserted` says which)
  int64_t& at(int64_t key, bool& inserted) {
    size_t i = detail::hash(key) & mask_;
    while (keys_[i] != detail::kEmpty && keys_[i] != key) i = (i + 1) & mask_;
    inserted = keys_[i] == detail::kEmpty;
    keys_[i] = key;
    return vals_[i];
  }
  const int64_t* find(int64_t key) const {
    if (key < 0) return nullptr;
    size_t i = detail::hash(key) & mask_;
    while (keys_[i] != detail::kEmpty) {
      if (keys_[i] == key) return &vals_[i];
      i = (i + 1) & mask_;
    }
    return nullptr;
  }

 private:
  std::vector<int64_t> keys_, vals_;
  size_t mask_ = 0;
};

}  // namespace lidar3d
