#include "voice_interface/Intent.hpp"

std::cout << "=== Intent 使用示例 ===" << std::endl;

    // 创建Intent并设置Action
    voice_interface::Intent intent;
    intent.setAction("control_light");

    // 添加各种类型的数据
    intent.putString("name", "客厅灯")
        .putBoolean("enabled", true)
        .putInt("brightness", 80)
        .putDouble("price", 299.99)
        .putFloat("color_temp", 4000.0f)
        .putNull("optional_param");

    // 添加数组数据
    intent.putIntArray("rgb_values", {255, 128, 64});
    intent.putStringArray("tags", {"light", "living_room", "smart"});

    // 打印Intent信息
    std::cout << "\n1. Intent信息:" << std::endl;
    std::cout << "   Action: " << intent.getAction().value_or("none") << std::endl;

    // 获取并打印各个字段
    auto name = intent.getString("name");
    auto enabled = intent.getBoolean("enabled");
    auto brightness = intent.getInt("brightness");
    auto price = intent.getDouble("price");
    auto colorTemp = intent.getFloat("color_temp");
    auto isNull = intent.isNull("optional_param");
    auto rgb = intent.getIntArray("rgb_values");
    auto tags = intent.getStringArray("tags");

    if (name)
        std::cout << "   名称: " << *name << std::endl;
    if (enabled)
        std::cout << "   状态: " << (*enabled ? "开启" : "关闭") << std::endl;
    if (brightness)
        std::cout << "   亮度: " << *brightness << std::endl;
    if (price)
        std::cout << "   价格: " << *price << std::endl;
    if (colorTemp)
        std::cout << "   色温: " << *colorTemp << std::endl;
    std::cout << "   optional_param isNull: " << (isNull ? "true" : "false") << std::endl;

    if (rgb)
    {
        std::cout << "   RGB值: ";
        for (size_t i = 0; i < rgb->size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << (*rgb)[i];
        }
        std::cout << std::endl;
    }

    if (tags)
    {
        std::cout << "   标签: ";
        for (size_t i = 0; i < tags->size(); ++i)
        {
            if (i > 0)
                std::cout << ", ";
            std::cout << (*tags)[i];
        }
        std::cout << std::endl;
    }

    // 转换为Intent字符串
    std::string intentStr = intent.toIntentString();
    std::cout << "\n2. Intent字符串: " << intentStr << std::endl;

    // 从字符串解析
    voice_interface::Intent parsed;
    if (parsed.parseFromIntentString(intentStr))
    {
        std::cout << "\n3. 解析成功!" << std::endl;
        std::cout << "   解析后的Action: " << parsed.getAction().value_or("none") << std::endl;

        // 验证解析后的数据
        auto parsedName = parsed.getString("name");
        auto parsedEnabled = parsed.getBoolean("enabled");
        auto parsedBrightness = parsed.getInt("brightness");
        auto price = intent.getDouble("price");

        if (parsedName)
            std::cout << "   解析后的名称: " << *parsedName << std::endl;
        if (parsedEnabled)
            std::cout << "   解析后的状态: " << (*parsedEnabled ? "开启" : "关闭") << std::endl;
        if (parsedBrightness)
            std::cout << "   解析后的亮度: " << *parsedBrightness << std::endl;
        if (price)
            std::cout << "   解析后的price: " << *price << std::endl;
    }

    // 转换为URI格式
    std::string uri = intent.toUri();
    std::cout << "\n4. URI格式: " << uri << std::endl;
    voice_interface::Intent parsed2;
    if (parsed2.parseFromUri(uri))
    {
        std::cout << "\n5. 解析成功!" << std::endl;
        std::cout << "   解析后的Action: " << parsed.getAction().value_or("none") << std::endl;

        // 验证解析后的数据
        auto parsedName = parsed2.getString("name");
        auto parsedEnabled = parsed2.getBoolean("enabled");
        auto parsedBrightness = parsed2.getInt("brightness");

        if (parsedName)
            std::cout << "   解析后的名称: " << *parsedName << std::endl;
        if (parsedEnabled)
            std::cout << "   解析后的状态: " << (*parsedEnabled ? "开启" : "关闭") << std::endl;
        if (parsedBrightness)
            std::cout << "   解析后的亮度: " << *parsedBrightness << std::endl;
    }
