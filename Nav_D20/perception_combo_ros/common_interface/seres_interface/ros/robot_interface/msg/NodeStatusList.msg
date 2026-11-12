# 所有节点的状态数组（嵌套NodeStatus类型）
NodeStatus[] nodes                     # 每个元素描述一个节点的状态
builtin_interfaces/Time report_time    # 本次上报的时间戳
string reporter_id                     # 上报来源ID（如监控模块名称）