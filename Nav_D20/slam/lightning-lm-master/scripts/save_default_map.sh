ros2 service call /lightning/save_map lightning/srv/SaveMap "{map_id: 86b0feda-ab99-414b-95fa-6cc6788d1998}"

ros2 action send_goal /nav/mapping nav2_interface/action/Mapping '{map_name: "office_building", map_uuid: ""}' --feedback
ros2 topic pub --once /map/switch std_msgs/msg/String '{data: "/data/map/robot_service/ad68517c-e845-42a4-b5dd-a6b7c6d2af13"}'


ros2 topic pub --once /map/switch std_msgs/msg/String '{data: "data/ad68517c-e845-42a4-b5dd-a6b7c6d2af13"}'
ros2 action send_goal /nav/relocation nav2_interface/action/Relocation '{pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, map_uuid: "./data/ad68517c-e845-42a4-b5dd-a6b7c6d2af13"}'  --feedback

ros2 action send_goal /nav/relocation nav2_interface/action/Relocation '{pose: {position: {x: -636.735, y: 736.71, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, map_uuid: "9ddffd42-d8c9-40e0-8931-0bc29e07c7c7"}'  --feedback

ros2 topic pub --once /map/switch std_msgs/msg/String '{data: "data/0c3f8a2d-41eb-416d-8fca-aaad500e2be9"}'
ros2 action send_goal /nav/relocation nav2_interface/action/Relocation '{pose: {position: {x: -636.735, y: 736.71, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, map_uuid: "ad68517c-e845-42a4-b5dd-a6b7c6d2af13"}'  --feedback

ros2 action send_goal /nav/relocation nav2_interface/action/Relocation '{pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, map_uuid: "s122345"}'  --feedback
ros2 action send_goal /nav/relocation nav2_interface/action/Relocation '{pose: {position: {x: 0.0, y: 0.0, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}, map_uuid: "new_map"}'  --feedback


 ros2 action send_goal /nav/relocation nav2_interface/action/Relocation '{pose: {position: {x: -636.735, y: 736.71, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1}}, map_uuid: "9ddffd42-d8c9-40e0-8931-0bc29e07c7c7"}'  --feedback

 ros2 action send_goal /nav/relocation nav2_interface/action/Relocation '{pose: {position: {x: -175.93, y: 440.824, z: 0.0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1}}, map_uuid: "new_map_1"}'  --feedback

ros2 action cancel_goal <goal_id>


ros2 action cancel_goal /nav/mapping --all
ros2 action cancel_goal /nav/relocation --all