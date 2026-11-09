/*
 * @Descripttion:
 * @version:
 * @Author: ZhaoTian
 * @Date: 2023-05-26 16:38:07
 * @LastEditors: ZhaoTian
 * @LastEditTime: 2023-05-26 17:03:59
 */
#pragma once
#include "src/common/easysocket/DllHelper.h"
#include "src/common/easysocket/basesocket.h"
#include <functional>
#include <string>
#include <thread>
namespace common {
namespace device {
class EASYSOCKET_API UDPSocket : public BaseSocket {
public:
  std::function<void(std::string, std::string, std::uint16_t)>
      onMessageReceived;
  std::function<void(const char *, int, std::string, std::uint16_t)>
      onRawMessageReceived;

  explicit UDPSocket(bool useConnect = false, FDR_ON_ERROR, int socketId = -1);

  void SendTo(std::string message, std::string host, uint16_t port,
              FDR_ON_ERROR);
  void SendTo(const char *bytes, size_t byteslength, std::string host,
              uint16_t port, FDR_ON_ERROR);

  int Send(std::string message);
  int Send(const char *bytes, size_t byteslength);

  void Connect(std::string host, uint16_t port, FDR_ON_ERROR);
  void Connect(uint32_t ipv4, uint16_t port, FDR_ON_ERROR);

private:
  static void Receive(UDPSocket *udpSocket);
  static void ReceiveFrom(UDPSocket *udpSocket);
};
} // namespace socket
} // namespace common_base
