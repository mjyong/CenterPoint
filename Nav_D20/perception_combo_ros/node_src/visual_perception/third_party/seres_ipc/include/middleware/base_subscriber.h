#ifndef SUBSCRIBER_BASE_H
#define SUBSCRIBER_BASE_H
#include <memory>
#include <string>

namespace ros_ipc_com
{
    class SubscriberBase : public std::enable_shared_from_this<SubscriberBase>
    {
    public:
        SubscriberBase(const std::string &topic_name) : m_topic_name(topic_name) {};
        virtual ~SubscriberBase() = default;

        const std::string &get_topic_name() const
        {
            return m_topic_name;
        }

    private:
        std::string m_topic_name; // Topic name
    };
}
#endif