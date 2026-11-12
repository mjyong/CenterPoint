#ifndef ROS_IPC_LOG_H
#define ROS_IPC_LOG_H
#include <iostream>
#include <string>
#include <ctime>
#include <cstdarg>

namespace ros_ipc_com
{
    enum LogLevel
    {
        LOG_LEVEL_INFO = 0,
        LOG_LEVEL_WARNING,
        LOG_LEVEL_ERROR,
        LOG_LEVEL_FATAL
    };

    // 当前日志级别设置
    #ifndef CURRENT_LOG_LEVEL
    #define CURRENT_LOG_LEVEL LOG_LEVEL_INFO
    #endif

    class Logger
    {
    public:
        static std::string getTimeString()
        {
            time_t t = time(nullptr);
            char buf[64] = {0};

            struct tm *tm_info = localtime(&t);
            strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tm_info);
            return buf;
        }

        template <LogLevel level>
        static void log(const char *file, int line, const char *func, const char *fmt, ...)
        {
            (void)file;
             
            if (level < CURRENT_LOG_LEVEL)
                return;

            const char *levelStr = "";
            switch (level)
            {
            case LOG_LEVEL_INFO:
                levelStr = "INFO";
                break;
            case LOG_LEVEL_WARNING:
                levelStr = "WARNING";
                break;
            case LOG_LEVEL_ERROR:
                levelStr = "ERROR";
                break;
            case LOG_LEVEL_FATAL:
                levelStr = "FATAL";
                break;
            }

            char msg[1024] = {0};
            va_list args;
            va_start(args, fmt);
            vsnprintf(msg, sizeof(msg), fmt, args);
            va_end(args);

            // 当前使用标准输出
            std::ostream& out = (level >= LOG_LEVEL_ERROR) ? std::cerr : std::cout;
            out << "[" << getTimeString() << "][" << levelStr << "][" << func << ":" << line << "] " << msg << std::endl;

            // 未来切换到glog的实现
            //google::LogMessage(file, line, static_cast<google::LogSeverity>(level)).stream() << msg;
        }
    };

} // namespace ros_ipc_com

// 日志宏定义
#define IPC_LOG_INFO(fmt, ...) ros_ipc_com::Logger::log<ros_ipc_com::LOG_LEVEL_INFO>(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define IPC_LOG_WARNING(fmt, ...) ros_ipc_com::Logger::log<ros_ipc_com::LOG_LEVEL_WARNING>(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define IPC_LOG_ERROR(fmt, ...) ros_ipc_com::Logger::log<ros_ipc_com::LOG_LEVEL_ERROR>(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)
#define IPC_LOG_FATAL(fmt, ...) ros_ipc_com::Logger::log<ros_ipc_com::LOG_LEVEL_FATAL>(__FILE__, __LINE__, __func__, fmt, ##__VA_ARGS__)

#endif // ROS_IPC_LOG_H