/*
 * @Descripttion: 
 * @version: 
 * @Author: ZhaoTian
 * @Date: 2023-05-26 16:38:07
 * @LastEditors: ZhaoTian
 * @LastEditTime: 2023-05-26 17:03:35
 */
#include "src/common/easysocket/udpserver.h"
namespace common {
namespace device {
UDPServer::UDPServer() {}

void UDPServer::Bind(int port, std::function<void(int, std::string)> onError) {
  int opt = 1;
  setsockopt(this->sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  this->Bind("0.0.0.0", port, onError);
}

void UDPServer::Bind(std::string IPv4, std::uint16_t port,
                     std::function<void(int, std::string)> onError) {
  if (inet_pton(AF_INET, IPv4.c_str(), &this->address.sin_addr) <= 0) {
    onError(errno, "Invalid address. Address type not supported.");
    return;
  }

  this->address.sin_family = AF_INET;
  this->address.sin_port = htons(port);
  int opt = 1;
  setsockopt(this->sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  if (bind(this->sock, (const sockaddr *)&this->address,
           sizeof(this->address)) < 0) {
    onError(errno, "Cannot bind the socket.");
    return;
  }
}
} // namespace socket
} // namespace common_base