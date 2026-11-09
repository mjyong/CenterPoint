#!/usr/bin/env python3
"""调用 /perception/perception_detect，把返回的 jpeg 存到 ./tmp。"""

import sys
from datetime import datetime
from pathlib import Path

import rclpy
from rclpy.node import Node
from perception_interface.srv import PerceptionEvent

SERVICE_NAME = "/perception/perception_detect"
OUT_DIR = Path("./tmp")
# 14 彩色（建筑物外观），15 彩色（路灯），13 红外，19 货车停放连拍
DEFAULT_EVENT_ID = 14


def main():
    event_id = DEFAULT_EVENT_ID
    if len(sys.argv) > 1:
        event_id = int(sys.argv[1])

    rclpy.init()
    node = Node("ptz_grab_once")
    client = node.create_client(PerceptionEvent, SERVICE_NAME)
    if not client.wait_for_service(timeout_sec=10.0):
        print(f"service not available: {SERVICE_NAME}", file=sys.stderr)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    request = PerceptionEvent.Request()
    request.percept.event_id = event_id
    future = client.call_async(request)
    rclpy.spin_until_future_complete(node, future)
    result = future.result()
    node.destroy_node()
    rclpy.shutdown()

    if result is None:
        print("empty response", file=sys.stderr)
        return 1

    # 新接口把图放在 event_info，连拍时一张图一个元素。
    # 这台狗上的 26100901 直接放在 response.object。
    if getattr(result, "event_info", None):
        infos = list(result.event_info)
    elif getattr(result, "object", None) is not None:
        infos = [result]
    else:
        print("empty response", file=sys.stderr)
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved = 0
    for index, info in enumerate(infos):
        payload = bytes(info.object.data)
        print(f"format: {info.object.format}  bytes: {len(payload)}")
        if info.object.format != "jpeg":
            print(payload.decode("utf-8", "replace"))
            return 1
        name = f"ptz_grab_{event_id}_{stamp}.jpg"
        if len(infos) > 1:
            name = f"ptz_grab_{event_id}_{stamp}_{index}.jpg"
        out_path = OUT_DIR / name
        out_path.write_bytes(payload)
        print(f"saved {out_path}")
        saved += 1
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
