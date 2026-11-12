#ifndef ROS_WRAPPER_H
#define ROS_WRAPPER_H

#include <string>
#include <memory>
#include <map>
#include <thread>
#include <functional>
#include "base_publisher.h"
#include "base_subscriber.h"
#include "base_timer.h"
#include "ros_ipc_log.h"
#include "rclcpp/rclcpp.hpp"

namespace ros_ipc_com
{
    template <typename T>
    class RosPublisher : public PublisherBase<T>
    {
    public:
        RosPublisher(const std::string &topic_name, int queue_size, std::shared_ptr<rclcpp::Node> &node)
            : PublisherBase<T>(topic_name),
              m_node(node)
        {
            m_publisher = m_node->create_publisher<T>(topic_name, queue_size);
            if (!m_publisher)
            {
                IPC_LOG_ERROR("failed to create ROS2 publisher for topic: %s", topic_name.c_str());
            }
        }
        ~RosPublisher()
        {
            IPC_LOG_INFO("topic [ %s ] publisher destroyed", this->get_topic_name().c_str());
        };

        int send_message(const T &msg)
        {
            m_publisher->publish(msg);
            return 0;
        }

    private:
        // ROS publisher
        typename rclcpp::Publisher<T>::SharedPtr m_publisher;
        std::shared_ptr<rclcpp::Node> m_node;
    };

    template <typename T>
    class RosSubscriber : public SubscriberBase
    {
    public:
        RosSubscriber(const std::string &topic_name,
                      int queue_size,
                      std::function<void(const std::shared_ptr<T> &)> callback,
                      std::shared_ptr<rclcpp::Node> &node)
            : SubscriberBase(topic_name),
              m_callback(callback),
              m_node(node)
        {

            auto ros_message_callback = [this](const typename T::SharedPtr msg)
            {
                m_callback(msg);
            };

            m_subscriber = m_node->create_subscription<T>(topic_name, queue_size, ros_message_callback);
            if (!m_subscriber)
            {
                IPC_LOG_ERROR("failed to create ROS2 subscriber for topic: %s", topic_name.c_str());
            }
        }

        ~RosSubscriber()
        {
            IPC_LOG_INFO("topic [ %s ] publisher subscriber", this->get_topic_name().c_str());
        };

    private:
        // ROS subscriber
        std::function<void(const std::shared_ptr<T> &)> m_callback;
        typename rclcpp::Subscription<T>::SharedPtr m_subscriber;
        std::shared_ptr<rclcpp::Node> m_node;
    };

    class RosTimer : public TimerBase
    {
    public:
        RosTimer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback,
                 std::shared_ptr<rclcpp::Node> &node)
            : TimerBase(name, period, oneShot, callback),
              m_node(node) {}
        ~RosTimer() { IPC_LOG_WARNING("timer [ %s ] released", timerName().c_str()); }

        void start() override;
        void stop() override;
        void restart() override;
        bool is_active() override;
        std::chrono::milliseconds time_until_trigger() const override;

        void timeout_callback();

    private:
        rclcpp::TimerBase::SharedPtr m_timer;
        std::shared_ptr<rclcpp::Node> m_node;
    };

    class RosWrapper
    {
    public:
        RosWrapper() = default;
        ~RosWrapper() = default;

        /**
         * @brief Initialize the ROS node.
         * @return 0 on success, non-zero on failure.
         */
        int init(std::string node_name);

        /**
         * @brief Deinitialize the ROS node and clean up resources.
         */
        void deinit();

        /**
         * @brief Create a publisher for the specified topic.
         * @param topic_name The name of the topic.
         * @param queue_size The size of the message queue.
         * @return The publisher object.
         */
        template <typename T>
        std::shared_ptr<RosPublisher<T>> create_publisher(const std::string &topic_name, int queue_size = 10)
        {
            std::shared_ptr<RosPublisher<T>> publisher = std::make_shared<RosPublisher<T>>(topic_name, queue_size, m_node);
            return publisher;
        }

        /**
         * @brief Create a subscriber for the specified topic.
         * @param topic_name The name of the topic.
         * @param callback The callback function to handle received ros messages.
         * @return The subscriber object.
         */
        template <typename T>
        std::shared_ptr<RosSubscriber<T>> create_subscriber(const std::string &topic_name, int queue_size, std::function<void(const std::shared_ptr<T> &)> callback)
        {
            std::shared_ptr<RosSubscriber<T>> subscriber = std::make_shared<RosSubscriber<T>>(topic_name, queue_size, callback, m_node);
            return subscriber;
        }

        /**
         * @brief create a timer
         * @param period timeout of the timer (milliseconds)
         * @param callback
         */
        std::shared_ptr<TimerBase> create_timer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback);

        /**
         * @brief spin node
         */
        void run();

        void exit();

    private:
        std::shared_ptr<rclcpp::Node> m_node; // ROS node handle
    };
} // namespace ros_ipc_com
#endif // ROS_WRAPPER_H