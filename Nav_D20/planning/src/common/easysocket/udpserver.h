/*
 * @Descripttion: 
 * @version: 
 * @Author: ZhaoTian
 * @Date: 2023-05-26 16:38:07
 * @LastEditors: ZhaoTian
 * @LastEditTime: 2023-05-26 17:03:43
 */
#pragma once

#include "src/common/easysocket/DllHelper.h"

#include "src/common/easysocket/udpsocket.h"
#include <string>
namespace common {
namespace device {
class EASYSOCKET_API UDPServer : public UDPSocket {
public:
  UDPServer();

  void Bind(int port, FDR_ON_ERROR);
  void Bind(std::string IPv4, std::uint16_t port, FDR_ON_ERROR);
};
} // namespace socket
} // namespace common_base
