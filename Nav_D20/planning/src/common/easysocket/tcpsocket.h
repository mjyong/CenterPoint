/*
 * @Descripttion: 
 * @version: 
 * @Author: ZhaoTian
 * @Date: 2023-05-26 16:38:07
 * @LastEditors: ZhaoTian
 * @LastEditTime: 2023-05-26 17:03:13
 */
#pragma once

#include "src/common/easysocket/DllHelper.h"

#include "src/common/easysocket/basesocket.h"
#include <functional>
#include <string>
#include <thread>
namespace common {
namespace device {
class EASYSOCKET_API TCPSocket : public BaseSocket {
public:
  // Event Listeners:
  std::function<void(std::string)> onMessageReceived;
  std::function<void(const char *, int)> onRawMessageReceived;
  std::function<void()> onSocketClosed;

  explicit TCPSocket(FDR_ON_ERROR, int socketId = -1);

  int Send(std::string message);
  int Send(const char *bytes, size_t byteslength);

  void Connect(
      std::string host, uint16_t port,
      std::function<void()> onConnected = []() {}, FDR_ON_ERROR);
  void Connect(
      uint32_t ipv4, uint16_t port, std::function<void()> onConnected = []() {},
      FDR_ON_ERROR);

  void Listen();

  void setAddressStruct(sockaddr_in addr);
  sockaddr_in getAddressStruct() const;

  void setTimeout(int seconds);
  int getTimeout() const;

private:
  static void Receive(TCPSocket *socket);

  int timeout = 5; // 5 seconds timeout default.
};
} // namespace socket
} // namespace common_base
