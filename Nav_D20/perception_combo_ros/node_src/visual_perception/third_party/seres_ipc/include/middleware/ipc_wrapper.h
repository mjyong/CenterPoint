#ifndef IPC_WARAPPER_H
#define IPC_WARAPPER_H

#include <map>
#include <string>
#include <cstdint>
#include <thread>
#include <functional>
#include <memory>
#include <mutex>
#include <iostream>
#include <atomic>
#include <queue>
#include <condition_variable>
#include "base_publisher.h"
#include "base_subscriber.h"
#include "timer_manager.h"
#include "ros_ipc_log.h"
#include "SERES_IPC_Init.h"

#define FRAME_ID_SIZE sizeof(uint32_t)
namespace ros_ipc_com
{
    template <typename T>
    class IpcPublisher;
    struct LockGuard
    {
        enum LockType
        {
            READ_LOCK = 0,
            WRITE_LOCK
        };

        uint32_t buf_id;
        uint32_t buf_idx;
        bool active;

        LockGuard(uint32_t id, uint32_t idx, LockType type) : buf_id(id), buf_idx(idx), active(true)
        {
            // 锁定内存
            if (type == READ_LOCK)
            {
                seres_read_lock(buf_id, buf_idx);
            }
            else
            {
                seres_write_lock(buf_id, buf_idx);
            }
        }

        ~LockGuard()
        {
            if (active)
            {
                // 释放内存锁
                seres_release_lock(buf_id, buf_idx);
            }
        }

        void release()
        {
            active = false; // 不再由 LockGuard 管理
        }
    };

    struct BufferItem
    {
        uint32_t buffer_id;
        std::shared_ptr<void> buffer_ptr;
    };

    class IPCCallbackInterface
    {
    public:
        IPCCallbackInterface() = default;
        virtual ~IPCCallbackInterface() = default;
        virtual void call(const std::shared_ptr<void> &raw_msg) const = 0;
    };

    template <typename T>
    class IPCSubscriberCallback : public IPCCallbackInterface
    {
    public:
        using callback_t = std::function<void(const std::shared_ptr<T> &)>;
        callback_t m_callback;

        IPCSubscriberCallback(const callback_t &cb) : m_callback(cb) {}
        ~IPCSubscriberCallback() override = default;
        void call(const std::shared_ptr<void> &raw_msg) const override
        {
            m_callback(std::static_pointer_cast<T>(raw_msg));
        }
    };

    /**
     * @brief IPC IpcSubscriber class
     */
    class IpcSubscriber : public SubscriberBase
    {
    public:
        IpcSubscriber(const std::string &topic_name, const std::shared_ptr<IPCCallbackInterface> &callback)
            : SubscriberBase(topic_name), m_subscriber_callback(callback) {}

        ~IpcSubscriber()
        {
            IPC_LOG_WARNING("topic [ %s ] subscriber destoryed", this->get_topic_name().c_str());
        };

        void IpcMsgCallback(const std::shared_ptr<void> &raw_msg)
        {
            m_subscriber_callback->call(raw_msg);
        }

    private:
        std::shared_ptr<IPCCallbackInterface> m_subscriber_callback; // Subscriber callback
    };

    class IpcWrapper : public std::enable_shared_from_this<IpcWrapper>
    {
    public:
        IpcWrapper();

        ~IpcWrapper();

        /**
         * @brief Initialize the IPC wrapper.
         * @return 0 on success, non-zero on failure.
         */
        int init(const std::string &process_name);

        void deinit();
        /**
         * @brief Create a publisher for the specified topic.
         * @param topic_name The name of the topic.
         * @return The buffer ID for the topic.
         */
        template <typename T>
        std::shared_ptr<IpcPublisher<T>> create_publisher(const std::string &topic_name)
        {
            std::shared_ptr<IpcPublisher<T>> publisher = nullptr;
            if (m_init_flag != 0)
            {
                IPC_LOG_ERROR("IPC has not initialized yet!");
                return publisher;
            }

            int buf_id = -1;
            if ((buf_id = seres_get_buf_id_by_name(topic_name.c_str())) > 0)
            {
                // buf size should >= sizeof(T) + FRAME_ID_SIZE
                if (IPC_BUF_Sets[buf_id].buf_size >= sizeof(T) + FRAME_ID_SIZE)
                {
                    publisher = std::make_shared<IpcPublisher<T>>(topic_name, buf_id, shared_from_this());
                }
                else
                {
                    IPC_LOG_ERROR("buf_size not enough for topic: %s ,at least %d Bytes", topic_name.c_str(), sizeof(T) + FRAME_ID_SIZE);
                }
            }
            else
            {
                IPC_LOG_ERROR("failed to get buffer id for topic: %s", topic_name.c_str());
            }
            return publisher;
        }

        /**
         * @brief Create a subscriber for the specified topic.
         * @param topic_name The name of the topic.
         * @param callback The callback function to be called when a message is received.
         * @return The subscriber object.
         */
        template <typename T>
        std::shared_ptr<IpcSubscriber> create_subscriber(const std::string &topic_name, typename IPCSubscriberCallback<T>::callback_t callback)
        {
            std::shared_ptr<IpcSubscriber> subscriber = nullptr;
            if (m_init_flag != 0)
            {
                IPC_LOG_ERROR("IPC has not initialized yet!");
                return subscriber;
            }

            int buf_id = 0;
            if ((buf_id = seres_get_buf_id_by_name(topic_name.c_str())) > 0)
            {
                // buf size should >= sizeof(T) + FRAME_ID_SIZE
                if (IPC_BUF_Sets[buf_id].buf_size >= sizeof(T) + FRAME_ID_SIZE)
                {
                    auto ipc_callback = std::make_shared<IPCSubscriberCallback<T>>(callback);
                    subscriber = std::make_shared<IpcSubscriber>(topic_name, ipc_callback);
                    add_ipc_subscriber(buf_id, subscriber);
                }
                else
                {
                    IPC_LOG_ERROR("buf_size not enough for topic: %s ,at least %d Bytes", topic_name.c_str(), sizeof(T) + FRAME_ID_SIZE);
                }
            }
            else
            {
                IPC_LOG_ERROR("failed to get buffer id for topic: %s", topic_name.c_str());
            }

            return subscriber;
        }

        /**
         * @brief create a timer
         * @param period timeout of the timer (milliseconds)
         * @param callback
         */
        std::shared_ptr<TimerBase> create_timer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback);

        /**
         * @brief Send a message to the specified buffer.
         * @param buf_id The buffer ID to send the message to.
         * @param frame_id
         * @param msg The message to send.
         * @return The result of the send operation.
         */
        template <typename T>
        int send_message(int buf_id, uint32_t frame_id, const T &msg)
        {
            uint16_t buf_idx = frame_id % IPC_BUF_Sets[buf_id].buf_num;
            {
                LockGuard write_lock_guard(buf_id, buf_idx, LockGuard::WRITE_LOCK); // 创建临时写锁对象
                memcpy(IPC_BUF_Sets[buf_id].buf_addr[buf_idx], &frame_id, FRAME_ID_SIZE);
                memcpy(static_cast<char *>(IPC_BUF_Sets[buf_id].buf_addr[buf_idx]) + FRAME_ID_SIZE, &msg, sizeof(T));
            }

            return seres_send_ipc_msg(buf_id, frame_id, buf_idx);
        }

        void run();

        void exit();

#ifdef UNIT_TEST
    public:
#else
    private:
#endif
        void handle_wait_message();
        void start_subscriber_thread();
        void add_ipc_subscriber(uint32_t buf_id, const std::shared_ptr<IpcSubscriber> &subscriber);
        void ipc_message_callback(uint32_t buf_id, const std::shared_ptr<void> &msg);
        void ipc_message_notify(uint32_t buf_id, uint32_t frame_id, uint32_t buf_idx);

#ifdef UNIT_TEST
    public:
#else
    private:
#endif
        int m_init_flag;
        std::mutex m_subscriber_mutex;                                                  // Mutex for subscriber map access
        std::map<uint32_t, std::vector<std::weak_ptr<IpcSubscriber>>> m_subscriber_map; // Subscriber map buffer id to subscriber
        std::mutex m_buffer_mutex;
        std::queue<BufferItem> m_buffer_queue;
        std::atomic<bool> m_running_flag;              // running flag
        std::shared_ptr<TimerManager> m_timer_manager; // timer manager
    };

    /**
     * @brief IPC Publisher class
     * @tparam T Message type
     */
    template <typename T>
    class IpcPublisher : public PublisherBase<T>
    {
    public:
        IpcPublisher(const std::string &topic_name, int buf_id, const std::shared_ptr<IpcWrapper> &ipc_wrapper)
            : PublisherBase<T>(topic_name),
              m_buf_id(buf_id),
              m_ipc_wrapper(ipc_wrapper) {}

        ~IpcPublisher()
        {
            IPC_LOG_WARNING("topic [ %s ] publisher destoryed", this->get_topic_name().c_str());
        };

        int send_message(const T &msg)
        {
            return m_ipc_wrapper->template send_message<T>(m_buf_id, get_frame_id(), msg);
        }

    private:
        uint32_t get_frame_id()
        {
            uint32_t frame_id = m_frame_id.fetch_add(1);
            return frame_id;
        }

    private:
        // IPC wrapper
        int m_buf_id; // Buffer ID
        std::atomic<uint32_t> m_frame_id;
        std::shared_ptr<IpcWrapper> m_ipc_wrapper;
    };

} // namespace ros_ipc_com
#endif // IPC_WARAPPER_H