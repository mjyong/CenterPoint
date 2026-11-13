# 语音内容定义
string type
string data
uint8[] binz

# type: 内容类型
#   text   - 纯文本内容
#   json   - JSON格式数据
#   intent - Intent对象，用于传递结构化参数
#   scheme - URI Scheme格式
#   status - 状态信息
#   binary - 二进制数据，配合binz字段使用
#   base64 - Base64编码的数据
#   function - 功能调用标识
#   error - 错误信息
# data: 内容数据
#   当type=binary时，此字段可为空，数据存储在binz中
#   其他类型时存储对应的文本数据，如文本内容、JSON字符串、Intent序列化URI、Scheme URI、状态描述、Base64字符串、功能名称、错误描述等
# binz: 二进制数据存储
#   当type=binary时，存储实际的二进制数据
#   其他类型时为空数组