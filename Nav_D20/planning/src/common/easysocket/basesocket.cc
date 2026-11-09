/*
 * @Descripttion:
 * @version:
 * @Author: ZhaoTian
 * @Date: 2023-05-26 16:38:07
 * @LastEditors: ZhaoTian
 * @LastEditTime: 2023-05-26 17:02:38
 */
#include "src/common/easysocket/basesocket.h"

#include <cerrno>
namespace common {
namespace device {
std::string BaseSocket::ipToString(sockaddr_in addr) {
  char ip[INET_ADDRSTRLEN];
  inet_ntop(AF_INET, &(addr.sin_addr), ip, INET_ADDRSTRLEN);

  return std::string(ip);
}

BaseSocket::BaseSocket(std::function<void(int, std::string)> onError,
                       SocketType sockType, int socketId)
    : sock(socketId) {
  if (socketId < 0) {
    if ((this->sock = socket(AF_INET, sockType, 0)) < 0) {
      onError(errno, "Socket creating error.");
    }
  }
}

void BaseSocket::Close() {
  if (isClosed)
    return;

  isClosed = true;
  close(this->sock);
}

std::string BaseSocket::remoteAddress() { return ipToString(this->address); }

int BaseSocket::remotePort() { return ntohs(this->address.sin_port); }
} // namespace device
} // namespace common