#ifndef _SERES_STRING_H_
#define _SERES_STRING_H_

#include "stddef.h"
#include "stdint.h"
#include <cstring>
#include <string>

namespace Seres {
template <uint32_t N> class string {
public:
  string() {
  }

  string(const std::string &str) {
    if (str.length() >= N) {
      std::__throw_out_of_range("out of range.");
    }

    set_data(str.c_str(), str.length());
  }

  string(const string<N> &other) {
    set_data(other.m_data, other.m_len);
  }

  string(string<N> &&other) {
    set_data(other.m_data, other.m_len);
    other.m_len = 0;
  }

  string &operator=(const std::string &str) {
    if (str.length() >= N) {
      std::__throw_out_of_range("out of range.");
    }

    set_data(str.c_str(), str.length());

    return *this;
  }

  string<N> &operator=(const string<N> &other) {
    set_data(other.m_data, other.m_len);

    return *this;
  }

  string<N> &operator=(string<N> &&other) {
    set_data(other.m_data, other.m_len);
    other.m_len = 0;

    return *this;
  }

  bool operator==(const char *str) const {
    if (str == nullptr)
      return false;

    if (strlen(str) != m_len) {
      return false;
    }

    return (memcmp(m_data, str, m_len) == 0);
  }

  bool operator==(const std::string &str) const {
    if (str.length() != m_len) {
      return false;
    }

    return (memcmp(m_data, str.c_str(), m_len) == 0);
  }

  bool operator==(const string<N> &str) const {
    if (str.m_len != m_len) {
      return false;
    }

    return (memcmp(m_data, str.m_data, m_len) == 0);
  }

  bool operator!=(const char *str) const {
    if (str == nullptr)
      return true;

    if (strlen(str) != m_len) {
      return true;
    }

    return (memcmp(m_data, str, m_len) != 0);
  }

  bool operator!=(const std::string &str) const {
    if (str.length() != m_len) {
      return true;
    }

    return (memcmp(m_data, str.c_str(), m_len) != 0);
  }

  bool operator!=(const string<N> &str) const {
    if (str.m_len != m_len) {
      return true;
    }

    return (memcmp(m_data, str.m_data, m_len) != 0);
  }

  ~string() {
  }

  void clear() {
    m_len = 0;
  }

  uint32_t length() {
    return m_len;
  }

  char *c_str() {
    return m_data;
  }

  const char *c_str() const {
    return m_data;
  }

private:
  void set_data(const char *data, uint32_t len) {
    memcpy(m_data, data, len);
    m_data[len] = '\0';
    m_len = len;
  }

private:
  uint32_t m_len = 0;
  char m_data[N];
};
} // namespace Seres

#endif