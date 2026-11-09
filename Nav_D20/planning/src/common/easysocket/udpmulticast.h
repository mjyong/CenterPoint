/*
 * @Descripttion: 
 * @version: 
 * @Author: ZhaoTian
 * @Date: 2023-05-26 16:38:07
 * @LastEditors: ZhaoTian
 * @LastEditTime: 2023-05-26 17:03:28
 */
#pragma once

#include "src/common/easysocket/DllHelper.h"

#include "src/common/easysocket/udpsocket.h"
#include <string.h>
#include <string>
#include <vector>
namespace common {
namespace device {
class EASYSOCKET_API UDPMulticast : public UDPSocket {
public:
  UDPMulticast();

  void BindRecv(int port, FDR_ON_ERROR);
  void Bind(std::string IPv4, std::uint16_t port, FDR_ON_ERROR);
  void BindDevice(std::string device, FDR_ON_ERROR);
  void SetRecvMulticastIp(const std::vector<std::string> &multicastIp);
  void SetSendMulticastIp(const std::vector<std::string> &multicastIp);
  void BindSend(int port, std::function<void(int, std::string)> onError);
  void SendData(const char *bytes, size_t byteslength, std::string host,
                std::function<void(int, std::string)> onError);

private:
  std::vector<std::string> m_RecvMulticastIp;
  std::vector<std::string> m_SendMulticastIp;
  std::uint16_t m_MulticastPort;
  int m_sendfd;
};
} // namespace socket
} // namespace common_base
