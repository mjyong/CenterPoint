#include "src/common/easysocket/udpmulticast.h"
namespace common {
namespace device {
UDPMulticast::UDPMulticast() {}

void UDPMulticast::BindRecv(int port,
                            std::function<void(int, std::string)> onError) {
  int opt = 1;
  setsockopt(this->sock, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  this->Bind("0.0.0.0", port, onError);
  m_MulticastPort = port;
}

void UDPMulticast::Bind(std::string IPv4, std::uint16_t port,
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
  for (auto ip : m_RecvMulticastIp) {
    struct ip_mreq mreq;
    mreq.imr_multiaddr.s_addr = inet_addr(ip.c_str());
    mreq.imr_interface.s_addr = inet_addr("192.168.1.102");
    if (setsockopt(this->sock, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq,
                   sizeof(mreq)) < 0) {
      std::string err_str = "Cannot setsockopt IP_ADD_MEMBERSHIP. " + ip;
      onError(errno, err_str);
      return;
    }
  }
}

void UDPMulticast::SetRecvMulticastIp(
    const std::vector<std::string> &multicastIp) {
  m_RecvMulticastIp = multicastIp;
}

void UDPMulticast::SetSendMulticastIp(
    const std::vector<std::string> &multicastIp) {
  m_SendMulticastIp = multicastIp;
}

void UDPMulticast::BindDevice(std::string device,
                              std::function<void(int, std::string)> onError) {
  struct ip_mreqn ip_addr;
  ip_addr.imr_ifindex = if_nametoindex(device.c_str());
  if (setsockopt(this->sock, IPPROTO_IP, IP_MULTICAST_IF, &ip_addr,
                 sizeof(ip_addr)) < 0) {
    onError(errno, "Cannot setsockopt IP_MULTICAST_IF.");
  }
}

void UDPMulticast::BindSend(int port,
                            std::function<void(int, std::string)> onError) {
  if ((m_sendfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
    onError(errno, "Socket creating error.");
  }
  sockaddr_in address;
  if (inet_pton(AF_INET, "192.168.1.102", &address.sin_addr) <= 0) {
    onError(errno, "Invalid address. Address type not supported.");
    return;
  }

  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  int opt = 1;
  setsockopt(m_sendfd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
  // int loop = 0; //TODO
  // if (setsockopt(m_sendfd, IPPROTO_IP, IP_MULTICAST_LOOP, &loop,
  // sizeof(loop)) < 0) {
  //     onError(errno, "Cannot setsockopt IP_MULTICAST_LOOP.");
  // }
  if (bind(m_sendfd, (const sockaddr *)&address, sizeof(address)) < 0) {
    onError(errno, "Cannot bind the socket.");
    return;
  }
  // for (auto ip : m_SendMulticastIp) {
  //     struct ip_mreq mreq;
  //     mreq.imr_multiaddr.s_addr = inet_addr(ip.c_str());
  //     mreq.imr_interface.s_addr = htonl(INADDR_ANY);
  //     if (setsockopt(m_sendfd, IPPROTO_IP, IP_ADD_MEMBERSHIP, &mreq,
  //     sizeof(mreq)) < 0) {
  //         std:: string err_str = "Cannot setsockopt IP_ADD_MEMBERSHIP. " +
  //         ip; onError(errno, err_str); return;
  //     }
  // }
}

void UDPMulticast::SendData(const char *bytes, size_t byteslength,
                            std::string host,
                            std::function<void(int, std::string)> onError) {
  sockaddr_in hostAddr;

  struct addrinfo hints, *res, *it;
  memset(&hints, 0, sizeof(hints));
  hints.ai_family = AF_INET;
  hints.ai_socktype = SOCK_DGRAM;

  int status;
  if ((status = getaddrinfo(host.c_str(), NULL, &hints, &res)) != 0) {
    onError(errno, "Invalid address." + std::string(gai_strerror(status)));
    return;
  }

  for (it = res; it != NULL; it = it->ai_next) {
    if (it->ai_family == AF_INET) {
      memcpy((void *)(&hostAddr), (void *)it->ai_addr, sizeof(sockaddr_in));
      break;
    }
  }

  freeaddrinfo(res);

  hostAddr.sin_port = htons(m_MulticastPort);
  hostAddr.sin_family = AF_INET;
  int ret = sendto(m_sendfd, bytes, byteslength, 0, (sockaddr *)&hostAddr,
                   sizeof(hostAddr));
  if (ret < 0) {
    onError(errno, "Cannot send message to the address.");
    return;
  }
}
} // namespace device
} // namespace common_base