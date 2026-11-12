#ifndef PUBLISHER_BASE_H
#define PUBLISHER_BASE_H
#include <memory>
#include <string>

namespace ros_ipc_com
{
    template <typename T>
    class PublisherBase : public std::enable_shared_from_this<PublisherBase<T>>
    {
    public:
        PublisherBase(const std::string &topic_name) : m_topic_name(topic_name) {};
        virtual ~PublisherBase() = default;

        const std::string &get_topic_name() const
        {
            return m_topic_name;
        }

        virtual int send_message(const T &msg) = 0;

    private:
        std::string m_topic_name; // Topic name
    };
}

#endif