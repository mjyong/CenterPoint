/*
 * @Descripttion:
 * @version:
 * @Author: ZhaoTian
 * @Date: 2023-05-26 16:38:07
 * @LastEditors: ZhaoTian
 * @LastEditTime: 2023-05-26 17:02:27
 */
#pragma once

#include "src/common/easysocket/DllHelper.h"

#if defined(__linux__) || defined(__APPLE__)
#include <arpa/inet.h>
#include <net/if.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#elif _WIN32
#include <winsock32.h>
#endif

#include <functional>
#include <string>

#define FDR_UNUSED(expr)                                                       \
  { (void)(expr); }
#define FDR_ON_ERROR                                                           \
  std::function<void(int, std::string)> onError =                              \
      [](int errorCode, std::string errorMessage) {                            \
        FDR_UNUSED(errorCode);                                                 \
        FDR_UNUSED(errorMessage)                                               \
      }
namespace common {
namespace device {
class EASYSOCKET_API BaseSocket {
protected:
  int sock = 0;

  static std::string ipToString(sockaddr_in addr);

public:
  const uint16_t BUFFER_SIZE = 0xFFFF;
  enum EASYSOCKET_API SocketType { TCP = SOCK_STREAM, UDP = SOCK_DGRAM };

  sockaddr_in address;
  bool isClosed = false;

  explicit BaseSocket(FDR_ON_ERROR, SocketType sockType = TCP,
                      int socketId = -1);

  virtual void Close();

  std::string remoteAddress();
  int remotePort();
  int fileDescriptor() const { return this->sock; }
};
} // namespace device
} // namespace common_base
