#ifndef BASE_TIMER_H
#define BASE_TIMER_H
#include <functional>
#include <chrono>
#include <string>

namespace ros_ipc_com
{
    class TimerBase
    {
    public:
        using TimerCallback = std::function<void()>;

        TimerBase(const std::string &name, int period, bool oneShot, TimerBase::TimerCallback callback) : m_timer_name(name),
                                                                                                          m_period(period),
                                                                                                          m_oneShot(oneShot),
                                                                                                          m_callback(callback) {}
        virtual ~TimerBase() = default;

        virtual void start() = 0;
        virtual void stop() = 0;
        //it cannot be called in the timer callback function
        virtual void restart() = 0;
        virtual bool is_active() = 0;
        virtual std::chrono::milliseconds time_until_trigger() const = 0;
        virtual const std::string& timerName() const {
            return m_timer_name;
        }

    protected:
        std::string m_timer_name;
        int m_period;
        bool m_oneShot;
        TimerBase::TimerCallback m_callback;
    };
}
#endif