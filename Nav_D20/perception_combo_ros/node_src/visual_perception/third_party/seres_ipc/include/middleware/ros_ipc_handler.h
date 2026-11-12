#ifndef ROS_IPC_HANDLER_H
#define ROS_IPC_HANDLER_H
#include <string>
#include <memory>
#include <mutex>
#include <functional>
#include <iostream>
#ifdef COM_IPC
#include "ipc_wrapper.h"
#else
#include "ros_wrapper.h"
#endif
namespace ros_ipc_com
{
      class RosIpcHandler
      {
      public:
            ~RosIpcHandler() = default;

            /**
             * @brief get the instace of the wrapper handler
             * @return the wrapper handler
             */
            static RosIpcHandler &get_instance();

            /**
             * @brief Initialize the IPC or ROS wrapper.
             * @param process_name The name of the process. for IPC, this is the process name; for ROS, this is the node name.
             * @return true on success, non-zero on failure.
             */
            int init(const std::string &process_name);

            /**
             * @brief Deinitialize the IPC or ROS wrapper.
             * @note if ROS wrapper is used, it will shutdown the ROS node and clean up rcl resources.
             */
            void deinit();

            /**
             * @brief Create a publisher for the specified topic.
             * @param topic_name The name of the topic.
             * @return The publisher object.
             */
            template <typename T>
            std::shared_ptr<PublisherBase<T>> create_publisher(const std::string &topic_name)
            {
                  std::shared_ptr<PublisherBase<T>> publisher = nullptr;

#ifdef COM_IPC
                  publisher = m_ipc_wrapper->create_publisher<T>(topic_name);
#else
                  publisher = m_ros_wrapper->create_publisher<T>(topic_name);
#endif
                  return publisher;
            }

            /**
             * @brief Create a subscriber for the specified topic.
             * @param topic_name The name of the topic.
             * @param callback The callback function to be called when a message is received.
             * @return The subscriber object.
             */
            template <typename T>
            std::shared_ptr<SubscriberBase> create_subscriber(const std::string &topic_name, std::function<void(const std::shared_ptr<T> &)> callback)
            {
                  std::shared_ptr<SubscriberBase> subscriber = nullptr;

#ifdef COM_IPC
                  subscriber = m_ipc_wrapper->create_subscriber<T>(topic_name, callback);
#else
                  subscriber = m_ros_wrapper->create_subscriber<T>(topic_name, 10, callback);
#endif
                  return subscriber;
            }

            /**
             * @brief Create a timer with a trigger period of the set period.
             * The trigger period includes the execution time of the callback function.
             * If the execution time of the callback function exceeds the period, it will be triggered immediately next time
             * @param name timer name
             * @param period timer period
             * @param oneShout trigger once
             * @param callback timer callback
             */
            std::shared_ptr<TimerBase> create_timer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback)
            {
#ifdef COM_IPC
                  return m_ipc_wrapper->create_timer(name, period, oneShot, callback);
#else
                  return m_ros_wrapper->create_timer(name, period, oneShot, callback);
#endif
            }

            /**
             * @brief process message
             */
            void run();

            /**
             * @brief exit
             */
            void exit();

      private:
            RosIpcHandler() = default;
            RosIpcHandler(const RosIpcHandler &) = delete;            // Disable copy constructor
            RosIpcHandler &operator=(const RosIpcHandler &) = delete; // Disable assignment operator
            RosIpcHandler(RosIpcHandler &&) = delete;                 // Disable move constructor
            RosIpcHandler &operator=(RosIpcHandler &&) = delete;      // Disable move assignment operator

      private:
#ifdef COM_IPC
            std::shared_ptr<IpcWrapper> m_ipc_wrapper; // IPC wrapper
#else
            std::shared_ptr<RosWrapper> m_ros_wrapper; // ROS wrapper
#endif
      };
} // namespace ros_ipc_com
#endif // ROS_IPC_HANDLER_H