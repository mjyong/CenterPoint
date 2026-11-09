/******************************************************************************
 * Copyright 2019 The Apollo Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *****************************************************************************/

/**
 * @file
 **/

#pragma once // 防止头文件重复包含

#include <utility> // 包含pair模板类
#include <vector>  // 包含vector容器

// #include "osqp/osqp.h" // 包含OSQP优化求解器库
#include <osqp.h>
namespace planning {
namespace algorithm {

/**
 * @brief 有限元位置偏差OSQP接口类
 *
 * 该类实现了基于有限元方法和位置偏差约束的路径平滑优化算法，
 * 使用OSQP求解器进行二次规划求解，用于生成平滑的车辆轨迹。
 */
class FemPosDeviationOsqpInterface {
public:
  // 默认构造函数和析构函数
  FemPosDeviationOsqpInterface() = default;
  virtual ~FemPosDeviationOsqpInterface() = default;

  // 设置参考点序列（x,y坐标对）
  void
  set_ref_points(const std::vector<std::pair<double, double>> &ref_points) {
    ref_points_ = ref_points;
  }

  // 设置每个参考点周围的允许偏差边界
  void set_bounds_around_refs(const std::vector<double> &bounds_around_refs) {
    bounds_around_refs_ = bounds_around_refs;
  }

  // 设置有限元位置偏差惩罚权重（控制平滑度）
  void set_weight_fem_pos_deviation(const double weight_fem_pos_deviation) {
    weight_fem_pos_deviation_ = weight_fem_pos_deviation;
  }

  // 设置路径长度惩罚权重（控制路径紧凑性）
  void set_weight_path_length(const double weight_path_length) {
    weight_path_length_ = weight_path_length;
  }

  // 设置参考点偏差惩罚权重（控制与参考点的接近程度）
  void set_weight_ref_deviation(const double weight_ref_deviation) {
    weight_ref_deviation_ = weight_ref_deviation;
  }

  // 设置OSQP求解器的最大迭代次数
  void set_max_iter(const int max_iter) { max_iter_ = max_iter; }

  // 设置求解时间限制（秒），0表示无限制
  void set_time_limit(const double time_limit) { time_limit_ = time_limit; }

  // 设置是否输出求解过程的详细信息
  void set_verbose(const bool verbose) { verbose_ = verbose; }

  // 设置是否使用缩放终止条件（提高数值稳定性）
  void set_scaled_termination(const bool scaled_termination) {
    scaled_termination_ = scaled_termination;
  }

  // 设置是否启用热启动（利用上一次求解结果加速收敛）
  void set_warm_start(const bool warm_start) { warm_start_ = warm_start; }

  /**
   * @brief 执行优化求解
   * @return 求解成功返回true，失败返回false
   */
  bool Solve();

  // 获取优化后的x坐标序列
  const std::vector<double> &opt_x() const { return x_; }

  // 获取优化后的y坐标序列
  const std::vector<double> &opt_y() const { return y_; }

private:
  /**
   * @brief 计算二次规划问题的核矩阵（Hessian矩阵）
   * @param P_data 核矩阵的非零元素值
   * @param P_indices 非零元素的行索引
   * @param P_indptr 列指针数组（CSC压缩存储格式）
   */
  void CalculateKernel(std::vector<double> *P_data,
                       std::vector<c_int> *P_indices,
                       std::vector<c_int> *P_indptr);

  /**
   * @brief 计算线性项（偏移量）
   * @param q 线性项系数向量
   */
  void CalculateOffset(std::vector<double> *q);

  /**
   * @brief 计算仿射约束条件
   * @param A_data 约束矩阵的非零元素值
   * @param A_indices 非零元素的行索引
   * @param A_indptr 列指针数组（CSC压缩存储格式）
   * @param lower_bounds 约束下界向量
   * @param upper_bounds 约束上界向量
   */
  void CalculateAffineConstraint(std::vector<double> *A_data,
                                 std::vector<c_int> *A_indices,
                                 std::vector<c_int> *A_indptr,
                                 std::vector<double> *lower_bounds,
                                 std::vector<double> *upper_bounds);

  /**
   * @brief 设置原始变量的热启动值
   * @param primal_warm_start 热启动向量
   */
  void SetPrimalWarmStart(std::vector<double> *primal_warm_start);

  /**
   * @brief 使用OSQP求解器进行优化
   * @param kernel_dim 优化变量维度
   * @param num_affine_constraint 约束条件数量
   * @param P_data 核矩阵数据
   * @param P_indices 核矩阵行索引
   * @param P_indptr 核矩阵列指针
   * @param A_data 约束矩阵数据
   * @param A_indices 约束矩阵行索引
   * @param A_indptr 约束矩阵列指针
   * @param lower_bounds 约束下界
   * @param upper_bounds 约束上界
   * @param q 线性项
   * @param primal_warm_start 热启动向量
   * @param data OSQP数据结构
   * @param work OSQP工作空间
   * @param settings OSQP求解器设置
   * @return 求解成功返回true，失败返回false
   */
  bool
  OptimizeWithOsqp(const size_t kernel_dim, const size_t num_affine_constraint,
                   std::vector<double> *P_data, std::vector<c_int> *P_indices,
                   std::vector<c_int> *P_indptr, std::vector<double> *A_data,
                   std::vector<c_int> *A_indices, std::vector<c_int> *A_indptr,
                   std::vector<double> *lower_bounds,
                   std::vector<double> *upper_bounds, std::vector<double> *q,
                   std::vector<double> *primal_warm_start, OSQPData *data,
                   OSQPWorkspace **work, OSQPSettings *settings);

private:
  // 参考点序列和偏差边界
  std::vector<std::pair<double, double>> ref_points_; // 参考点(x,y)坐标序列
  std::vector<double> bounds_around_refs_; // 每个参考点的允许偏差范围

  // 优化目标函数中的权重系数
  double weight_fem_pos_deviation_ =
      1.0e5; // 有限元位置偏差权重（默认值较大，强调平滑性）
  double weight_path_length_ = 1.0; // 路径长度权重（控制路径紧凑性）
  double weight_ref_deviation_ =
      1.0; // 参考点偏差权重（控制与参考点的接近程度）

  // OSQP求解器参数设置
  int max_iter_ = 4000;            // 最大迭代次数（默认4000次）
  double time_limit_ = 0.0;        // 时间限制（0表示无限制）
  bool verbose_ = false;           // 是否输出详细信息（默认关闭）
  bool scaled_termination_ = true; // 是否使用缩放终止条件（默认开启）
  bool warm_start_ = true;         // 是否启用热启动（默认开启）

  // 优化问题维度定义
  int num_of_points_ = 0;    // 参考点数量
  int num_of_variables_ = 0; // 优化变量数量（每个点有x,y两个坐标）
  int num_of_constraints_ = 0; // 约束条件数量

  // 优化结果存储
  std::vector<double> x_; // 优化后的x坐标序列
  std::vector<double> y_; // 优化后的y坐标序列
};

} // namespace algorithm
} // namespace planning