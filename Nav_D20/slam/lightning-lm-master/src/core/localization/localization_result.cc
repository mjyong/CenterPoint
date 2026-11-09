//
// Created by xiang on 24-4-11.
//

#include "localization_result.h"
#include "core/lightning_math.hpp"

namespace lightning::loc {

geometry_msgs::msg::TransformStamped LocalizationResult::ToGeoMsg(const SE3& transform) const {
    geometry_msgs::msg::TransformStamped msg;
    msg.header.frame_id = "map";
    msg.header.stamp = math::FromSec(timestamp_);
    msg.child_frame_id = "base_link";

    // 应用变换
    SE3 transformed_pose = pose_ * transform.inverse();

    msg.transform.translation.x = transformed_pose.translation().x();
    msg.transform.translation.y = transformed_pose.translation().y();
    msg.transform.translation.z = transformed_pose.translation().z();

    msg.transform.rotation.x = transformed_pose.so3().unit_quaternion().x();
    msg.transform.rotation.y = transformed_pose.so3().unit_quaternion().y();
    msg.transform.rotation.z = transformed_pose.so3().unit_quaternion().z();
    msg.transform.rotation.w = transformed_pose.so3().unit_quaternion().w();

    return msg;
}
nav_msgs::msg::Odometry LocalizationResult::ToOdomMsg(const SE3& transform) const {
    nav_msgs::msg::Odometry msg;
    msg.header.frame_id = "map";
    msg.header.stamp = math::FromSec(timestamp_);
    msg.child_frame_id = "base_link";

    // 应用变换
    SE3 transformed_pose = pose_ * transform.inverse();

    msg.pose.pose.position.x = transformed_pose.translation().x();
    msg.pose.pose.position.y = transformed_pose.translation().y();
    msg.pose.pose.position.z = transformed_pose.translation().z();

    msg.pose.pose.orientation.x = transformed_pose.so3().unit_quaternion().x();
    msg.pose.pose.orientation.y = transformed_pose.so3().unit_quaternion().y();
    msg.pose.pose.orientation.z = transformed_pose.so3().unit_quaternion().z();
    msg.pose.pose.orientation.w = transformed_pose.so3().unit_quaternion().w();

    // 速度也需要变换
    Vec3d vel_w =  transform.so3()*(vel_b_ + omega_b_.cross(transform.translation()));
    msg.twist.twist.linear.x = vel_w[0];
    msg.twist.twist.linear.y = vel_w[1];
    msg.twist.twist.linear.z = vel_w[2];
    Vec3d omega_w = transform.so3()*omega_b_;
    // 角速度是体坐标系下的，不需要变换
    msg.twist.twist.angular.x = omega_w[0];
    msg.twist.twist.angular.y = omega_w[1];
    msg.twist.twist.angular.z = omega_w[2];

    return msg;
}
NavState LocalizationResult::ToNavState(const SE3& transform) const {
    NavState ret;
    ret.timestamp_ = timestamp_;
    ret.confidence_ = confidence_;
    
    // 应用变换
    SE3 transformed_pose = pose_*transform.inverse();
    ret.pos_ = transformed_pose.translation();
    ret.rot_ = transformed_pose.so3();
    ret.pose_is_ok_ = status_ == LocalizationStatus::GOOD;

    // 速度也需要相应变换
    ret.vel_ = transformed_pose.so3() * vel_b_ - omega_b_.cross(transform.translation());
    ret.omega_b_ = transform.so3()*omega_b_  ;
    return ret;
}

}  // namespace lightning::loc
