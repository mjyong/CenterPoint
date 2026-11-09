#ifndef GRID_MAP_DATA_POOL_H
#define GRID_MAP_DATA_POOL_H

#include <mutex>
#include "grid_map_common_type.h"

#define GRID_MAP_DECLARE_SINGLETON(classname)                                  \
 public:                                                                       \
  static classname *Instance() {                                               \
    static classname *instance = nullptr;                                      \
    if (!instance) {                                                           \
      static std::once_flag flag;                                              \
      std::call_once(flag,                                                     \
                     [&] { instance = new (std::nothrow) classname(); });       \
    }                                                                          \
    return instance;                                                           \
  }                                                                            \
                                                                               \
 private:                                                                      \
  classname(const classname &) = delete;                                       \
  classname &operator=(const classname &) = delete;

namespace grid_map {

class DataPool {
public:
    DataPool() = default;

    GridMapMainData* GetMainDataPtr() { return &main_data_; }
    GridMapMainData& GetMainDataRef() { return main_data_; }

public:
    GridMapMainData main_data_;

    GRID_MAP_DECLARE_SINGLETON(DataPool)
};

} // namespace grid_map

#endif // GRID_MAP_DATA_POOL_H
