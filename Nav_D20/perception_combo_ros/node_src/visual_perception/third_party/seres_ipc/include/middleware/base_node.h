#ifndef BASE_NODE_H
#define BASE_NODE_H
#include "ros_ipc_handler.h"
#include "timer_manager.h"
#include <queue>
#include <csignal>

namespace ros_ipc_com
{
    class CLoop
    {
    public:
        using TaskT = std::function<void()>;
        using TaskTPtr = std::shared_ptr<TaskT>;

        CLoop()
        {
            m_timer_manager = std::make_shared<TimerManager>();
        }
        virtual ~CLoop() = default;

        /**
         * @brief Create a timer with a trigger period of the set period.
         * The trigger period includes the execution time of the callback function.
         * If the execution time of the callback function exceeds the period, it will be triggered immediately next time
         */
        std::shared_ptr<TimerBase> CreateTimer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback)
        {
            return m_timer_manager->create_timer(name, period, oneShot, callback);
        }

        void AddTask(const TaskT &task)
        {
            std::lock_guard<std::mutex> lock(m_task_mutex);
            m_task_que.push(std::make_shared<TaskT>(task));
            m_timer_manager->wake_up();
        }

        virtual void run()
        {
            m_thread_running = true;
            do
            {
                HandleWaitTask();
                m_timer_manager->check_and_wait();

            } while (m_thread_running);
        }

        virtual void exit()
        {
            m_thread_running = false;
            m_timer_manager->wake_up();
        }

    private:
        void HandleWaitTask()
        {
            std::queue<TaskTPtr> task_que;
            {
                std::lock_guard<std::mutex> lock(m_task_mutex);
                task_que = std::move(m_task_que);
            }

            while (!task_que.empty())
            {
                TaskTPtr &taskptr = task_que.front();
                (*taskptr)();
                task_que.pop();
            }
        }

    private:
        std::atomic<bool> m_thread_running;
        std::mutex m_task_mutex;
        std::queue<TaskTPtr> m_task_que;
        std::shared_ptr<TimerManager> m_timer_manager;
    };

    class ThreadLoop : public CLoop
    {
    public:
        void run() override
        {
            m_thread = std::thread(
                [this]()
                {
                    CLoop::run();
                });
        }

        void exit() override
        {
            CLoop::exit();
            if (m_thread.joinable())
            {
                m_thread.join();
            }
        }

    private:
        std::thread m_thread;
    };

    class NodeBase
    {
    public:
        NodeBase(const std::string &node_name)
        {
            if (RosIpcHandler::get_instance().init(node_name) != 0)
            {
                throw std::runtime_error("initailization failed");
            }

            InstallSignalHandler(SIGINT);
            InstallSignalHandler(SIGTERM);
        }

        virtual ~NodeBase()
        {
            RosIpcHandler::get_instance().deinit();
        }

        /**
         * @brief Create a publisher for the specified topic.
         * @param topic_name The name of the topic.
         * @return The publisher object.
         */
        template <typename T>
        std::shared_ptr<PublisherBase<T>> CreatePublisher(const std::string &topic_name)
        {
            return RosIpcHandler::get_instance().create_publisher<T>(topic_name);
        }

        /**
         * @brief Create a subscriber for the specified topic.
         * @param topic_name The name of the topic.
         * @param callback The callback function to be called when a message is received.
         * @return The subscriber object.
         */
        template <typename T>
        std::shared_ptr<SubscriberBase> createSubscriber(const std::string &topic_name, std::function<void(const std::shared_ptr<T> &)> callback)
        {
            return RosIpcHandler::get_instance().create_subscriber<T>(topic_name, callback);
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
        std::shared_ptr<TimerBase> CreateTimer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback)
        {
            return RosIpcHandler::get_instance().create_timer(name, period, oneShot, callback);
        }

        /**
         * @brief process message
         */
        void run()
        {
            RosIpcHandler::get_instance().run();
        }

        /**
         * @brief exit
         */
        void exit()
        {
            RosIpcHandler::get_instance().exit();
        }

    private:
        static void InstallSignalHandler(int signum);
        static void SignalHandler(int signum, siginfo_t *siginfo, void *context);

    private:
        static struct sigaction m_sigint_oldaction;
        static struct sigaction m_sigterm_oldaction;
    };
} // namespace ros_ipc_com
#endif // BASE_NODE_H