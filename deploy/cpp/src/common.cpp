#include "dogperc/common.hpp"

namespace dogperc {

const std::array<const char*, kNumClasses> kClasses = {"vehicle", "pedestrian", "cyclist"};
const std::array<const char*, kNumSubclasses> kSubclasses = {"car", "truck", "bus", "pedestrian", "cyclist"};

int class_index(const std::string& name) {
  for (int i = 0; i < kNumClasses; ++i)
    if (name == kClasses[i]) return i;
  return -1;
}

int subclass_index(const std::string& name) {
  for (int i = 0; i < kNumSubclasses; ++i)
    if (name == kSubclasses[i]) return i;
  return -1;
}

int default_subclass(int label) {
  if (label < 0 || label >= kNumClasses) return -1;
  return subclass_index(kClasses[label]);
}

}  // namespace dogperc
