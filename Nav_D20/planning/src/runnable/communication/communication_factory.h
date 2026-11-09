/*
 * @Author: zhaotian tian.zhao@seres.cn
 * @Date: 2024-02-22 19:33:03
 * @LastEditors: zhaotian tian.zhao@seres.cn
 * @LastEditTime: 2024-02-23 13:46:03
 */
#pragma once
#include "src/common/util/factory.h"
#include "src/runnable/communication/communication_base.h"
#include <memory>
using planning::common::util::Factory;
using namespace planning::communication;

namespace planning {
class CommunicationFactory {

public:
  CommunicationFactory();
  virtual ~CommunicationFactory() = default;
  std::unique_ptr<CommunicationBase>
  CreateCommunication(const CommunicationConf::Conf &conf);

private:
  static Factory<
      CommunicationConf::Type, CommunicationBase,
      CommunicationBase *(*)(const CommunicationConf::Conf &conf),
      std::unordered_map<
          CommunicationConf::Type,
          CommunicationBase *(*)(const CommunicationConf::Conf &conf),
          std::hash<int>>>
      communication_factory_;
};
} // namespace planning