#ifndef TIMER_MANAGER_H
#define TIMER_MANAGER_H
#include <memory>
#include <chrono>
#include <vector>
#include <set>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include "base_timer.h"

namespace ros_ipc_com
{
    class TimerManager;
    class CTimer : public TimerBase, public std::enable_shared_from_this<CTimer>
    {
    public:
        using CTimerPtr = std::weak_ptr<CTimer>;

        struct CompareTimer
        {
            bool operator()(const CTimerPtr &lhand, const CTimerPtr &rhand)
            {
                auto ltimer = lhand.lock();
                auto rtimer = rhand.lock();

                if (!ltimer && !rtimer)  // 两个都无效时，通过原始指针地址排序（确保不对称性）
                {
                    return &lhand < &rhand; // 用指针本身的地址作为唯一标识
                }
                else if (!ltimer) // 只有左无效：左排前面 便于移除
                {
                    return true;
                }
                else if (!rtimer) // 只有右无效：右排前面 便于移除
                {
                    return false;
                }
                else
                {
                    if (ltimer->m_time_point == rtimer->m_time_point)
                    {
                        return ltimer.get() < rtimer.get(); // 如果触发时间相同，按内存地址排序
                    }
                    else
                    {
                        return ltimer->m_time_point < rtimer->m_time_point;
                    }
                }
            }
        };

        CTimer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback,
               const std::shared_ptr<TimerManager> &timer_manager);
        ~CTimer();

        void start() override;

        void stop() override;

        void restart() override;

        bool is_active() override;

        std::chrono::milliseconds time_until_trigger() const override;

        void update_expiration();

        void timeout_callback();

    private:
        bool m_active;
        std::chrono::time_point<std::chrono::steady_clock> m_time_point; // trigger timer point of steady clock
        std::shared_ptr<TimerManager> m_timer_manager;
    };

    /**
     * TimeManager
     */
    class TimerManager : public std::enable_shared_from_this<TimerManager>
    {
    public:
        TimerManager();
        ~TimerManager() = default;

        /**
         * @brief Create a timer with a trigger period of the set period.
         * The trigger period includes the execution time of the callback function.
         * If the execution time of the callback function exceeds the period, it will be triggered immediately next time
         */
        std::shared_ptr<TimerBase> create_timer(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback);

        void update_timer(const std::shared_ptr<CTimer> &new_timer);

        void check_and_wait();

        void wake_up();
#ifdef UNIT_TEST
    public:
#else
    private:
#endif
        std::mutex m_timer_mutex; // timers mutex
        std::set<CTimer::CTimerPtr, CTimer::CompareTimer> m_timer_set;
        std::mutex m_cv_mutex;                // mutex for condition variable
        std::condition_variable m_waiting_cv; // timer waiting condition variable
        std::atomic<bool> m_wake_up_flag;     // indicats that a wake up signal has been received
    };

} // namespace ros_ipc_com
#endif // TIMER_MANAGER_H