/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-06 10:58:52
 * @LastEditors: zhaotian tian.zhao@seres.cn
 * @LastEditTime: 2024-02-06 10:59:15
 */
#pragma once
#include <iostream>
#include <memory>
#include <mutex>

#ifndef BASE_DECLARE_SINGLETON
#define BASE_DECLARE_SINGLETON(classname)                                      \
public:                                                                        \
  static classname *Instance() {                                               \
    static classname *instance = nullptr;                                      \
    if (!instance) {                                                           \
      static std::once_flag flag;                                              \
      std::call_once(flag,                                                     \
                     [&] { instance = new (std::nothrow) classname(); });      \
    }                                                                          \
    return instance;                                                           \
  }                                                                            \
                                                                               \
private:                                                                       \
  classname(const classname &) = delete;                                       \
  classname &operator=(const classname &) = delete;
#endif