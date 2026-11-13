#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <sstream>
#include <iostream>
#include <regex>
#include <algorithm>
#include <cctype>
#include <iomanip>
#include <memory>
#include <string_view>

namespace voice_interface
{

    /**
     * @brief 性能优化的Android风格Intent实现
     *
     * 优化特性：
     * - 使用unordered_map实现O(1)查找
     * - 减少字符串拷贝
     * - 预分配内存
     * - 使用string_view避免临时对象
     * - 明确的类型命名避免重载歧义
     * - 支持初始化列表
     */
    class Intent
    {
    public:
        // ==================== 构造函数 ====================
        Intent()
        {
            // 预分配空间
            reserve(16);
        }

        explicit Intent(const std::string &action) : action_(action)
        {
            reserve(16);
        }

        Intent(const std::string &action, const std::string &data)
            : action_(action), data_(data)
        {
            reserve(16);
        }

        // 移动构造函数
        Intent(Intent &&other) noexcept
            : action_(std::move(other.action_)), data_(std::move(other.data_)), type_(std::move(other.type_)), categories_(std::move(other.categories_)), component_pkg_(std::move(other.component_pkg_)), component_class_(std::move(other.component_class_)), extras_string_(std::move(other.extras_string_)), extras_string_null_(std::move(other.extras_string_null_)), extras_boolean_(std::move(other.extras_boolean_)), extras_int_(std::move(other.extras_int_)), extras_long_(std::move(other.extras_long_)), extras_float_(std::move(other.extras_float_)), extras_double_(std::move(other.extras_double_)), extras_uri_(std::move(other.extras_uri_)), extras_component_(std::move(other.extras_component_)), extras_int_array_(std::move(other.extras_int_array_)), extras_long_array_(std::move(other.extras_long_array_)), extras_float_array_(std::move(other.extras_float_array_)), extras_string_array_(std::move(other.extras_string_array_))
        {
        }

        // 拷贝构造函数
        Intent(const Intent &) = default;

        // 移动赋值
        Intent &operator=(Intent &&other) noexcept
        {
            if (this != &other)
            {
                action_ = std::move(other.action_);
                data_ = std::move(other.data_);
                type_ = std::move(other.type_);
                categories_ = std::move(other.categories_);
                component_pkg_ = std::move(other.component_pkg_);
                component_class_ = std::move(other.component_class_);
                extras_string_ = std::move(other.extras_string_);
                extras_string_null_ = std::move(other.extras_string_null_);
                extras_boolean_ = std::move(other.extras_boolean_);
                extras_int_ = std::move(other.extras_int_);
                extras_long_ = std::move(other.extras_long_);
                extras_float_ = std::move(other.extras_float_);
                extras_double_ = std::move(other.extras_double_);
                extras_uri_ = std::move(other.extras_uri_);
                extras_component_ = std::move(other.extras_component_);
                extras_int_array_ = std::move(other.extras_int_array_);
                extras_long_array_ = std::move(other.extras_long_array_);
                extras_float_array_ = std::move(other.extras_float_array_);
                extras_string_array_ = std::move(other.extras_string_array_);
            }
            return *this;
        }

        // 拷贝赋值
        Intent &operator=(const Intent &) = default;

        ~Intent() = default;

        // ==================== 核心方法 ====================

        /**
         * @brief 设置Action（后面的会覆盖前面的）
         */
        Intent &setAction(const std::string &action)
        {
            action_ = action;
            return *this;
        }

        /**
         * @brief 移动版本的setAction，减少拷贝
         */
        Intent &setAction(std::string &&action)
        {
            action_ = std::move(action);
            return *this;
        }

        /**
         * @brief 获取Action
         */
        std::optional<std::string_view> getAction() const noexcept
        {
            return action_.empty() ? std::nullopt : std::optional<std::string_view>(action_);
        }

        /**
         * @brief 设置数据URI
         */
        Intent &setData(const std::string &data)
        {
            data_ = data;
            return *this;
        }

        Intent &setData(std::string &&data)
        {
            data_ = std::move(data);
            return *this;
        }

        std::optional<std::string_view> getData() const noexcept
        {
            return data_.empty() ? std::nullopt : std::optional<std::string_view>(data_);
        }

        /**
         * @brief 设置MIME类型
         */
        Intent &setType(const std::string &type)
        {
            type_ = type;
            return *this;
        }

        Intent &setType(std::string &&type)
        {
            type_ = std::move(type);
            return *this;
        }

        std::optional<std::string_view> getType() const noexcept
        {
            return type_.empty() ? std::nullopt : std::optional<std::string_view>(type_);
        }

        /**
         * @brief 同时设置数据和MIME类型
         */
        Intent &setDataAndType(const std::string &data, const std::string &type)
        {
            data_ = data;
            type_ = type;
            return *this;
        }

        /**
         * @brief 添加Category
         */
        Intent &addCategory(const std::string &category)
        {
            categories_.push_back(category);
            return *this;
        }

        Intent &addCategory(std::string &&category)
        {
            categories_.push_back(std::move(category));
            return *this;
        }

        const std::vector<std::string> &getCategories() const noexcept
        {
            return categories_;
        }

        bool hasCategory(std::string_view category) const noexcept
        {
            return std::find_if(categories_.begin(), categories_.end(),
                                [category](const std::string &c)
                                { return c == category; }) != categories_.end();
        }

        /**
         * @brief 设置组件名称
         */
        Intent &setComponent(const std::string &pkg, const std::string &cls)
        {
            component_pkg_ = pkg;
            component_class_ = cls;
            return *this;
        }

        Intent &setComponent(std::string_view component)
        {
            size_t slash_pos = component.find('/');
            if (slash_pos != std::string_view::npos)
            {
                component_pkg_ = std::string(component.substr(0, slash_pos));
                component_class_ = std::string(component.substr(slash_pos + 1));
            }
            else
            {
                component_pkg_ = std::string(component);
                component_class_.clear();
            }
            return *this;
        }

        std::optional<std::string> getComponent() const
        {
            if (component_pkg_.empty())
            {
                return std::nullopt;
            }
            if (component_class_.empty())
            {
                return component_pkg_;
            }
            return component_pkg_ + "/" + component_class_;
        }

        // ==================== Extra数据方法（明确类型命名，避免歧义）====================

        /**
         * @brief 添加String类型Extra
         * @param key 键
         * @param value 字符串值
         */
        Intent &putString(const std::string &key, const std::string &value)
        {
            removeKey(key);
            extras_string_.emplace(key, value);
            return *this;
        }

        /**
         * @brief 添加String类型Extra（移动版本）
         */
        Intent &putString(std::string &&key, std::string &&value)
        {
            removeKey(key);
            extras_string_.emplace(std::move(key), std::move(value));
            return *this;
        }

        /**
         * @brief 添加C风格字符串String类型Extra（便捷版本）
         */
        Intent &putString(const char *key, const char *value)
        {
            return putString(std::string(key), std::string(value));
        }

        /**
         * @brief 添加String Extra（值为null）
         */
        Intent &putNull(const std::string &key)
        {
            removeKey(key);
            extras_string_null_.insert(key);
            return *this;
        }

        Intent &putNull(std::string &&key)
        {
            removeKey(key);
            extras_string_null_.insert(std::move(key));
            return *this;
        }

        Intent &putNull(const char *key)
        {
            return putNull(std::string(key));
        }

        /**
         * @brief 添加boolean类型Extra
         */
        Intent &putBoolean(const std::string &key, bool value)
        {
            removeKey(key);
            extras_boolean_.emplace(key, value);
            return *this;
        }

        Intent &putBoolean(const char *key, bool value)
        {
            return putBoolean(std::string(key), value);
        }

        /**
         * @brief 添加int类型Extra
         */
        Intent &putInt(const std::string &key, int value)
        {
            removeKey(key);
            extras_int_.emplace(key, value);
            return *this;
        }

        Intent &putInt(const char *key, int value)
        {
            return putInt(std::string(key), value);
        }

        /**
         * @brief 添加long类型Extra
         */
        Intent &putLong(const std::string &key, long value)
        {
            removeKey(key);
            extras_long_.emplace(key, value);
            return *this;
        }

        Intent &putLong(const char *key, long value)
        {
            return putLong(std::string(key), value);
        }

        /**
         * @brief 添加float类型Extra
         */
        Intent &putFloat(const std::string &key, float value)
        {
            removeKey(key);
            extras_float_.emplace(key, value);
            return *this;
        }

        Intent &putFloat(const char *key, float value)
        {
            return putFloat(std::string(key), value);
        }

        /**
         * @brief 添加double类型Extra
         */
        Intent &putDouble(const std::string &key, double value)
        {
            removeKey(key);
            extras_double_.emplace(key, value);
            return *this;
        }

        Intent &putDouble(const char *key, double value)
        {
            return putDouble(std::string(key), value);
        }

        /**
         * @brief 添加URI类型Extra
         */
        Intent &putUri(const std::string &key, const std::string &value)
        {
            removeKey(key);
            extras_uri_.emplace(key, value);
            return *this;
        }

        Intent &putUri(const char *key, const char *value)
        {
            return putUri(std::string(key), std::string(value));
        }

        /**
         * @brief 添加ComponentName类型Extra
         */
        Intent &putComponent(const std::string &key, const std::string &pkg, const std::string &cls)
        {
            removeKey(key);
            extras_component_.emplace(key, pkg + "/" + cls);
            return *this;
        }

        Intent &putComponent(const char *key, const char *pkg, const char *cls)
        {
            return putComponent(std::string(key), std::string(pkg), std::string(cls));
        }

        /**
         * @brief 添加int数组类型Extra（vector版本）
         */
        Intent &putIntArray(const std::string &key, const std::vector<int> &values)
        {
            removeKey(key);
            extras_int_array_.emplace(key, values);
            return *this;
        }

        Intent &putIntArray(const std::string &key, std::vector<int> &&values)
        {
            removeKey(key);
            extras_int_array_.emplace(key, std::move(values));
            return *this;
        }

        Intent &putIntArray(const char *key, const std::vector<int> &values)
        {
            return putIntArray(std::string(key), values);
        }

        /**
         * @brief 添加int数组类型Extra（初始化列表版本）
         */
        Intent &putIntArray(const std::string &key, std::initializer_list<int> values)
        {
            return putIntArray(key, std::vector<int>(values));
        }

        Intent &putIntArray(const char *key, std::initializer_list<int> values)
        {
            return putIntArray(std::string(key), std::vector<int>(values));
        }

        /**
         * @brief 添加long数组类型Extra（vector版本）
         */
        Intent &putLongArray(const std::string &key, const std::vector<long> &values)
        {
            removeKey(key);
            extras_long_array_.emplace(key, values);
            return *this;
        }

        Intent &putLongArray(const std::string &key, std::vector<long> &&values)
        {
            removeKey(key);
            extras_long_array_.emplace(key, std::move(values));
            return *this;
        }

        Intent &putLongArray(const char *key, const std::vector<long> &values)
        {
            return putLongArray(std::string(key), values);
        }

        /**
         * @brief 添加long数组类型Extra（初始化列表版本）
         */
        Intent &putLongArray(const std::string &key, std::initializer_list<long> values)
        {
            return putLongArray(key, std::vector<long>(values));
        }

        Intent &putLongArray(const char *key, std::initializer_list<long> values)
        {
            return putLongArray(std::string(key), std::vector<long>(values));
        }

        /**
         * @brief 添加float数组类型Extra（vector版本）
         */
        Intent &putFloatArray(const std::string &key, const std::vector<float> &values)
        {
            removeKey(key);
            extras_float_array_.emplace(key, values);
            return *this;
        }

        Intent &putFloatArray(const std::string &key, std::vector<float> &&values)
        {
            removeKey(key);
            extras_float_array_.emplace(key, std::move(values));
            return *this;
        }

        Intent &putFloatArray(const char *key, const std::vector<float> &values)
        {
            return putFloatArray(std::string(key), values);
        }

        /**
         * @brief 添加float数组类型Extra（初始化列表版本）
         */
        Intent &putFloatArray(const std::string &key, std::initializer_list<float> values)
        {
            return putFloatArray(key, std::vector<float>(values));
        }

        Intent &putFloatArray(const char *key, std::initializer_list<float> values)
        {
            return putFloatArray(std::string(key), std::vector<float>(values));
        }

        /**
         * @brief 添加String数组类型Extra（vector版本）
         */
        Intent &putStringArray(const std::string &key, const std::vector<std::string> &values)
        {
            removeKey(key);
            extras_string_array_.emplace(key, values);
            return *this;
        }

        Intent &putStringArray(const std::string &key, std::vector<std::string> &&values)
        {
            removeKey(key);
            extras_string_array_.emplace(key, std::move(values));
            return *this;
        }

        Intent &putStringArray(const char *key, const std::vector<std::string> &values)
        {
            return putStringArray(std::string(key), values);
        }

        /**
         * @brief 添加String数组类型Extra（初始化列表版本 - const char*）
         */
        Intent &putStringArray(const std::string &key, std::initializer_list<const char *> values)
        {
            std::vector<std::string> vec;
            vec.reserve(values.size());
            for (auto v : values)
            {
                vec.emplace_back(v);
            }
            return putStringArray(key, std::move(vec));
        }

        Intent &putStringArray(const char *key, std::initializer_list<const char *> values)
        {
            return putStringArray(std::string(key), values);
        }

        /**
         * @brief 添加String数组类型Extra（初始化列表版本 - string_view）
         */
        Intent &putStringArray(const std::string &key, std::initializer_list<std::string_view> values)
        {
            std::vector<std::string> vec;
            vec.reserve(values.size());
            for (auto v : values)
            {
                vec.emplace_back(v);
            }
            return putStringArray(key, std::move(vec));
        }

        Intent &putStringArray(const char *key, std::initializer_list<std::string_view> values)
        {
            return putStringArray(std::string(key), values);
        }

        /**
         * @brief 添加String数组类型Extra（C数组版本）
         */
        template <size_t N>
        Intent &putStringArray(const std::string &key, const char *(&values)[N])
        {
            std::vector<std::string> vec;
            vec.reserve(N);
            for (size_t i = 0; i < N; ++i)
            {
                vec.emplace_back(values[i]);
            }
            return putStringArray(key, std::move(vec));
        }

        template <size_t N>
        Intent &putStringArray(const char *key, const char *(&values)[N])
        {
            return putStringArray(std::string(key), values);
        }

        // ==================== 获取Extra数据的方法 ====================

        std::optional<std::string_view> getString(std::string_view key) const noexcept
        {
            auto it = extras_string_.find(std::string(key));
            if (it != extras_string_.end())
            {
                return std::string_view(it->second);
            }
            return std::nullopt;
        }

        bool isNull(std::string_view key) const noexcept
        {
            return extras_string_null_.find(std::string(key)) != extras_string_null_.end();
        }

        std::optional<bool> getBoolean(std::string_view key) const noexcept
        {
            auto it = extras_boolean_.find(std::string(key));
            if (it != extras_boolean_.end())
            {
                return it->second;
            }
            return std::nullopt;
        }

        std::optional<int> getInt(std::string_view key) const noexcept
        {
            auto it = extras_int_.find(std::string(key));
            if (it != extras_int_.end())
            {
                return it->second;
            }
            return std::nullopt;
        }

        std::optional<long> getLong(std::string_view key) const noexcept
        {
            auto it = extras_long_.find(std::string(key));
            if (it != extras_long_.end())
            {
                return it->second;
            }
            return std::nullopt;
        }

        std::optional<float> getFloat(std::string_view key) const noexcept
        {
            auto it = extras_float_.find(std::string(key));
            if (it != extras_float_.end())
            {
                return it->second;
            }
            return std::nullopt;
        }

        std::optional<double> getDouble(std::string_view key) const noexcept
        {
            auto it = extras_double_.find(std::string(key));
            if (it != extras_double_.end())
            {
                return it->second;
            }
            return std::nullopt;
        }

        std::optional<std::string_view> getUri(std::string_view key) const noexcept
        {
            auto it = extras_uri_.find(std::string(key));
            if (it != extras_uri_.end())
            {
                return std::string_view(it->second);
            }
            return std::nullopt;
        }

        std::optional<std::string_view> getComponent(std::string_view key) const noexcept
        {
            auto it = extras_component_.find(std::string(key));
            if (it != extras_component_.end())
            {
                return std::string_view(it->second);
            }
            return std::nullopt;
        }

        const std::vector<int> *getIntArray(std::string_view key) const noexcept
        {
            auto it = extras_int_array_.find(std::string(key));
            if (it != extras_int_array_.end())
            {
                return &(it->second);
            }
            return nullptr;
        }

        const std::vector<long> *getLongArray(std::string_view key) const noexcept
        {
            auto it = extras_long_array_.find(std::string(key));
            if (it != extras_long_array_.end())
            {
                return &(it->second);
            }
            return nullptr;
        }

        const std::vector<float> *getFloatArray(std::string_view key) const noexcept
        {
            auto it = extras_float_array_.find(std::string(key));
            if (it != extras_float_array_.end())
            {
                return &(it->second);
            }
            return nullptr;
        }

        const std::vector<std::string> *getStringArray(std::string_view key) const noexcept
        {
            auto it = extras_string_array_.find(std::string(key));
            if (it != extras_string_array_.end())
            {
                return &(it->second);
            }
            return nullptr;
        }

        /**
         * @brief 检查是否存在某个key
         */
        bool hasKey(std::string_view key) const noexcept
        {
            std::string key_str(key);
            return extras_string_.count(key_str) ||
                   extras_string_null_.count(key_str) ||
                   extras_boolean_.count(key_str) ||
                   extras_int_.count(key_str) ||
                   extras_long_.count(key_str) ||
                   extras_float_.count(key_str) ||
                   extras_double_.count(key_str) ||
                   extras_uri_.count(key_str) ||
                   extras_component_.count(key_str) ||
                   extras_int_array_.count(key_str) ||
                   extras_long_array_.count(key_str) ||
                   extras_float_array_.count(key_str) ||
                   extras_string_array_.count(key_str);
        }

        // ==================== 序列化方法 ====================

        /**
         * @brief 转换为URI格式的字符串
         */
        std::string toUri() const
        {
            // 预分配足够大的空间
            std::string result;
            result.reserve(256);

            result += "intent://";

            if (!data_.empty())
            {
                result += extractPathFromData();
            }

            result += "#Intent;";

            // 使用临时stringstream减少重新分配
            std::ostringstream oss;

            if (!action_.empty())
            {
                oss << "action=" << urlEncode(action_) << ";";
            }
            if (!data_.empty())
            {
                oss << "data=" << urlEncode(data_) << ";";
            }
            if (!type_.empty())
            {
                oss << "type=" << urlEncode(type_) << ";";
            }
            for (const auto &category : categories_)
            {
                oss << "category=" << urlEncode(category) << ";";
            }
            if (!component_pkg_.empty())
            {
                if (component_class_.empty())
                {
                    oss << "component=" << urlEncode(component_pkg_) << ";";
                }
                else
                {
                    oss << "component=" << urlEncode(component_pkg_ + "/" + component_class_) << ";";
                }
            }

            // Extra参数
            for (const auto &[key, value] : extras_string_)
            {
                oss << "S." << key << "=" << urlEncode(value) << ";";
            }
            for (const auto &key : extras_string_null_)
            {
                oss << "S." << key << "=null;";
            }
            for (const auto &[key, value] : extras_boolean_)
            {
                oss << "B." << key << "=" << (value ? "true" : "false") << ";";
            }
            for (const auto &[key, value] : extras_int_)
            {
                oss << "I." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_long_)
            {
                oss << "L." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_float_)
            {
                oss << "F." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_double_)
            {
                oss << "D." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_uri_)
            {
                oss << "U." << key << "=" << urlEncode(value) << ";";
            }
            for (const auto &[key, value] : extras_component_)
            {
                oss << "C." << key << "=" << urlEncode(value) << ";";
            }

            // 数组类型
            for (const auto &[key, values] : extras_int_array_)
            {
                oss << "IA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << values[i];
                }
                oss << ";";
            }
            for (const auto &[key, values] : extras_long_array_)
            {
                oss << "LA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << values[i];
                }
                oss << ";";
            }
            for (const auto &[key, values] : extras_float_array_)
            {
                oss << "FA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << values[i];
                }
                oss << ";";
            }
            for (const auto &[key, values] : extras_string_array_)
            {
                oss << "SA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << urlEncode(values[i]);
                }
                oss << ";";
            }

            oss << "end";
            result += oss.str();
            return result;
        }

        /**
         * @brief 转换为Intent字符串
         */
        std::string toIntentString() const
        {
            std::ostringstream oss;

            // 预分配空间
            oss.str().reserve(256);

            // 核心参数作为Extra的一部分
            if (!action_.empty())
            {
                oss << "S.action=" << action_ << ";";
            }
            if (!data_.empty())
            {
                oss << "S.data=" << data_ << ";";
            }
            if (!type_.empty())
            {
                oss << "S.type=" << type_ << ";";
            }
            for (const auto &category : categories_)
            {
                oss << "S.category=" << category << ";";
            }
            if (!component_pkg_.empty())
            {
                oss << "S.component=" << component_pkg_;
                if (!component_class_.empty())
                {
                    oss << "/" << component_class_;
                }
                oss << ";";
            }

            // 所有Extra参数
            for (const auto &[key, value] : extras_string_)
            {
                oss << "S." << key << "=" << value << ";";
            }
            for (const auto &key : extras_string_null_)
            {
                oss << "S." << key << "=null;";
            }
            for (const auto &[key, value] : extras_boolean_)
            {
                oss << "B." << key << "=" << (value ? "true" : "false") << ";";
            }
            for (const auto &[key, value] : extras_int_)
            {
                oss << "I." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_long_)
            {
                oss << "L." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_float_)
            {
                oss << "F." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_double_)
            {
                oss << "D." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_uri_)
            {
                oss << "U." << key << "=" << value << ";";
            }
            for (const auto &[key, value] : extras_component_)
            {
                oss << "C." << key << "=" << value << ";";
            }

            // 数组类型
            for (const auto &[key, values] : extras_int_array_)
            {
                oss << "IA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << values[i];
                }
                oss << ";";
            }
            for (const auto &[key, values] : extras_long_array_)
            {
                oss << "LA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << values[i];
                }
                oss << ";";
            }
            for (const auto &[key, values] : extras_float_array_)
            {
                oss << "FA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << values[i];
                }
                oss << ";";
            }
            for (const auto &[key, values] : extras_string_array_)
            {
                oss << "SA." << key << "=";
                for (size_t i = 0; i < values.size(); ++i)
                {
                    if (i > 0)
                        oss << ",";
                    oss << values[i];
                }
                oss << ";";
            }

            return oss.str();
        }

        /**
         * @brief 从URI字符串解析Intent
         */
        bool parseFromUri(const std::string &uri)
        {
            clear();

            size_t intentPos = uri.find("#Intent;");
            if (intentPos == std::string::npos)
            {
                return false;
            }

            if (intentPos > 0)
            {
                size_t schemePos = uri.find("://");
                if (schemePos != std::string::npos && schemePos + 3 < intentPos)
                {
                    data_ = uri.substr(schemePos + 3, intentPos - schemePos - 3);
                }
            }

            size_t start = intentPos + 8;
            size_t end = uri.find(";end", start);
            if (end == std::string::npos)
            {
                return false;
            }

            std::string_view intentPart(uri.data() + start, end - start);
            return parseIntentPart(intentPart);
        }

        /**
         * @brief 从Intent字符串解析
         */
        bool parseFromIntentString(const std::string &intentStr)
        {
            clear();
            return parseIntentPart(std::string_view(intentStr));
        }

        /**
         * @brief 清空所有数据
         */
        void clear() noexcept
        {
            action_.clear();
            data_.clear();
            type_.clear();
            categories_.clear();
            component_pkg_.clear();
            component_class_.clear();

            extras_string_.clear();
            extras_string_null_.clear();
            extras_boolean_.clear();
            extras_int_.clear();
            extras_long_.clear();
            extras_float_.clear();
            extras_double_.clear();
            extras_uri_.clear();
            extras_component_.clear();
            extras_int_array_.clear();
            extras_long_array_.clear();
            extras_float_array_.clear();
            extras_string_array_.clear();
        }

        /**
         * @brief 预分配空间
         */
        void reserve(size_t capacity)
        {
            extras_string_.reserve(capacity);
            extras_string_null_.reserve(capacity);
            extras_boolean_.reserve(capacity);
            extras_int_.reserve(capacity);
            extras_long_.reserve(capacity);
            extras_float_.reserve(capacity);
            extras_double_.reserve(capacity);
            extras_uri_.reserve(capacity);
            extras_component_.reserve(capacity);
            extras_int_array_.reserve(capacity);
            extras_long_array_.reserve(capacity);
            extras_float_array_.reserve(capacity);
            extras_string_array_.reserve(capacity);
            categories_.reserve(capacity);
        }

    private:
        // ==================== 私有方法 ====================

        /**
         * @brief 快速移除key
         */
        void removeKey(std::string_view key) noexcept
        {
            std::string key_str(key);
            extras_string_.erase(key_str);
            extras_string_null_.erase(key_str);
            extras_boolean_.erase(key_str);
            extras_int_.erase(key_str);
            extras_long_.erase(key_str);
            extras_float_.erase(key_str);
            extras_double_.erase(key_str);
            extras_uri_.erase(key_str);
            extras_component_.erase(key_str);
            extras_int_array_.erase(key_str);
            extras_long_array_.erase(key_str);
            extras_float_array_.erase(key_str);
            extras_string_array_.erase(key_str);
        }

        /**
         * @brief 解析Intent部分（使用string_view避免拷贝）
         */
        bool parseIntentPart(std::string_view intentPart)
        {
            size_t pos = 0;
            while (pos < intentPart.length())
            {
                size_t nextPos = intentPart.find(';', pos);
                if (nextPos == std::string_view::npos)
                {
                    nextPos = intentPart.length();
                }

                std::string_view item = intentPart.substr(pos, nextPos - pos);
                if (!item.empty())
                {
                    size_t eqPos = item.find('=');
                    if (eqPos != std::string_view::npos)
                    {
                        std::string_view key = item.substr(0, eqPos);
                        std::string_view value = item.substr(eqPos + 1);

                        parseItem(key, value);
                    }
                }

                pos = nextPos + 1;
            }
            return true;
        }

        /**
         * @brief 解析单个项
         */
        void parseItem(std::string_view key, std::string_view value)
        {
            // 处理无前缀的核心字段（用于URI格式）
            if (key == "action")
            {
                action_ = urlDecode(value);
            }
            else if (key == "data")
            {
                data_ = urlDecode(value);
            }
            else if (key == "type")
            {
                type_ = urlDecode(value);
            }
            else if (key == "category")
            {
                categories_.emplace_back(urlDecode(value));
            }
            else if (key == "component")
            {
                setComponent(urlDecode(value));
            }
            // 处理带前缀的核心字段（用于Intent字符串格式）
            else if (key == "S.action")
            {
                action_ = urlDecode(value);
            }
            else if (key == "S.data")
            {
                data_ = urlDecode(value);
            }
            else if (key == "S.type")
            {
                type_ = urlDecode(value);
            }
            else if (key == "S.category")
            {
                categories_.emplace_back(urlDecode(value));
            }
            else if (key == "S.component")
            {
                setComponent(urlDecode(value));
            }
            // 处理带其他前缀的字段（如 S., B., I., L., F., D., U., C., IA., LA., FA., SA.）
            else if (key.find('.') != std::string_view::npos)
            {
                parseExtraItem(key, value);
            }
        }

        /**
         * @brief 解析Extra项
         */
        void parseExtraItem(std::string_view key, std::string_view value)
        {
            size_t dotPos = key.find('.');
            if (dotPos == std::string_view::npos)
                return;

            std::string_view type = key.substr(0, dotPos);
            std::string_view realKey = key.substr(dotPos + 1);

            removeKey(realKey);
            std::string key_str(realKey);

            // 处理各种类型
            if (type == "S") // String
            {
                if (value == "null")
                {
                    extras_string_null_.insert(std::string(key_str));
                }
                else
                {
                    extras_string_.emplace(std::move(key_str), urlDecode(value));
                }
            }
            else if (type == "B") // Boolean
            {
                extras_boolean_.emplace(std::move(key_str), value == "true");
            }
            else if (type == "I") // Integer
            {
                try
                {
                    extras_int_.emplace(std::move(key_str), std::stoi(std::string(value)));
                }
                catch (...)
                {
                    // 解析失败时忽略
                }
            }
            else if (type == "L") // Long
            {
                try
                {
                    extras_long_.emplace(std::move(key_str), std::stol(std::string(value)));
                }
                catch (...)
                {
                    // 解析失败时忽略
                }
            }
            else if (type == "F") // Float
            {
                try
                {
                    extras_float_.emplace(std::move(key_str), std::stof(std::string(value)));
                }
                catch (...)
                {
                    // 解析失败时忽略
                }
            }
            else if (type == "D") // Double
            {
                try
                {
                    extras_double_.emplace(std::move(key_str), std::stod(std::string(value)));
                }
                catch (...)
                {
                    // 解析失败时忽略
                }
            }
            else if (type == "U") // URI
            {
                extras_uri_.emplace(std::move(key_str), urlDecode(value));
            }
            else if (type == "C") // Component
            {
                extras_component_.emplace(std::move(key_str), urlDecode(value));
            }
            else if (type == "IA") // Integer Array
            {
                std::vector<int> array;
                array.reserve(8);
                size_t start = 0;
                size_t end;
                std::string value_str(value);
                while ((end = value_str.find(',', start)) != std::string::npos)
                {
                    try
                    {
                        array.push_back(std::stoi(value_str.substr(start, end - start)));
                    }
                    catch (...)
                    {
                        // 解析失败时忽略该项
                    }
                    start = end + 1;
                }
                if (start < value_str.length())
                {
                    try
                    {
                        array.push_back(std::stoi(value_str.substr(start)));
                    }
                    catch (...)
                    {
                        // 解析失败时忽略
                    }
                }
                extras_int_array_.emplace(std::move(key_str), std::move(array));
            }
            else if (type == "LA") // Long Array
            {
                std::vector<long> array;
                array.reserve(8);
                size_t start = 0;
                size_t end;
                std::string value_str(value);
                while ((end = value_str.find(',', start)) != std::string::npos)
                {
                    try
                    {
                        array.push_back(std::stol(value_str.substr(start, end - start)));
                    }
                    catch (...)
                    {
                        // 解析失败时忽略该项
                    }
                    start = end + 1;
                }
                if (start < value_str.length())
                {
                    try
                    {
                        array.push_back(std::stol(value_str.substr(start)));
                    }
                    catch (...)
                    {
                        // 解析失败时忽略
                    }
                }
                extras_long_array_.emplace(std::move(key_str), std::move(array));
            }
            else if (type == "FA") // Float Array
            {
                std::vector<float> array;
                array.reserve(8);
                size_t start = 0;
                size_t end;
                std::string value_str(value);
                while ((end = value_str.find(',', start)) != std::string::npos)
                {
                    try
                    {
                        array.push_back(std::stof(value_str.substr(start, end - start)));
                    }
                    catch (...)
                    {
                        // 解析失败时忽略该项
                    }
                    start = end + 1;
                }
                if (start < value_str.length())
                {
                    try
                    {
                        array.push_back(std::stof(value_str.substr(start)));
                    }
                    catch (...)
                    {
                        // 解析失败时忽略
                    }
                }
                extras_float_array_.emplace(std::move(key_str), std::move(array));
            }
            else if (type == "SA") // String Array
            {
                std::vector<std::string> array;
                array.reserve(8);
                size_t start = 0;
                size_t end;
                std::string value_str(value);
                while ((end = value_str.find(',', start)) != std::string::npos)
                {
                    array.emplace_back(urlDecode(value_str.substr(start, end - start)));
                    start = end + 1;
                }
                if (start < value_str.length())
                {
                    array.emplace_back(urlDecode(value_str.substr(start)));
                }
                extras_string_array_.emplace(std::move(key_str), std::move(array));
            }
        }

        /**
         * @brief 从data中提取路径部分
         */
        std::string extractPathFromData() const
        {
            if (data_.empty())
                return "";

            size_t schemePos = data_.find("://");
            if (schemePos != std::string::npos)
            {
                return data_.substr(schemePos + 3);
            }
            return data_;
        }

        /**
         * @brief URL解码（优化版本）
         */
        static std::string urlDecode(std::string_view encoded)
        {
            std::string result;
            result.reserve(encoded.length());

            for (size_t i = 0; i < encoded.length(); ++i)
            {
                if (encoded[i] == '%' && i + 2 < encoded.length())
                {
                    int value;
                    std::istringstream is(std::string(encoded.substr(i + 1, 2)));
                    if (is >> std::hex >> value)
                    {
                        result += static_cast<char>(value);
                        i += 2;
                    }
                    else
                    {
                        result += '%';
                    }
                }
                else if (encoded[i] == '+')
                {
                    result += ' ';
                }
                else
                {
                    result += encoded[i];
                }
            }
            return result;
        }

        /**
         * @brief URL编码（优化版本）
         */
        static std::string urlEncode(std::string_view decoded)
        {
            std::ostringstream result;
            for (unsigned char c : decoded)
            {
                if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~')
                {
                    result << c;
                }
                else if (c == ' ')
                {
                    result << '+';
                }
                else
                {
                    result << '%' << std::hex << std::setw(2) << std::setfill('0')
                           << static_cast<int>(c);
                }
            }
            return result.str();
        }

        // ==================== 私有成员变量 ====================

        // 核心参数
        std::string action_;
        std::string data_;
        std::string type_;
        std::vector<std::string> categories_;
        std::string component_pkg_;
        std::string component_class_;

        // Extra数据 - 使用unordered_map以获得O(1)查找
        std::unordered_map<std::string, std::string> extras_string_;
        std::unordered_set<std::string> extras_string_null_;
        std::unordered_map<std::string, bool> extras_boolean_;
        std::unordered_map<std::string, int> extras_int_;
        std::unordered_map<std::string, long> extras_long_;
        std::unordered_map<std::string, float> extras_float_;
        std::unordered_map<std::string, double> extras_double_;
        std::unordered_map<std::string, std::string> extras_uri_;
        std::unordered_map<std::string, std::string> extras_component_;
        std::unordered_map<std::string, std::vector<int>> extras_int_array_;
        std::unordered_map<std::string, std::vector<long>> extras_long_array_;
        std::unordered_map<std::string, std::vector<float>> extras_float_array_;
        std::unordered_map<std::string, std::vector<std::string>> extras_string_array_;
    };

} // namespace voice_interface