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

#include "src/algorithm/grid_path_planner/discrete_points_smoothing/fem_pos_deviation_osqp_interface.h"
#include "src/common/util/log.h"
#include <limits>

namespace planning {
namespace algorithm {

bool FemPosDeviationOsqpInterface::Solve() {
  // Sanity Check // 输入参数检查，检查参考点是否为空
  if (ref_points_.empty()) {
    AERROR << "reference points empty, solver early terminates";
    return false;
  }
  // 检查参考点和边界约束数量是否匹配
  if (ref_points_.size() != bounds_around_refs_.size()) {
    AERROR << "ref_points and bounds size not equal, solver early terminates";
    return false;
  }
  // 检查参考点数量是否足够（至少需要4个点）
  if (ref_points_.size() < 4) {
    AERROR << "ref_points size smaller than 4, solver early terminates";
    return false;
  }
  // 检查参考点数量是否超过int类型的最大值限制
  if (ref_points_.size() > std::numeric_limits<int>::max()) {
    AERROR << "ref_points size too large, solver early terminates";
    return false;
  }

  // 计算优化问题的维度
  // num_of_points_: 参考点的数量
  // num_of_variables_: 优化变量数量（每个点有x,y两个坐标，所以乘以2）
  // num_of_constraints_: 约束条件数量（等于变量数量）
  // Calculate optimization states definitions
  num_of_points_ = static_cast<int>(ref_points_.size());
  num_of_variables_ = num_of_points_ * 2;
  num_of_constraints_ = num_of_variables_;

  // 计算二次规划问题的核矩阵（Hessian矩阵）
  // P_data: 核矩阵的非零元素值
  // P_indices: 非零元素的行索引
  // P_indptr: 列指针数组（CSC格式）
  // Calculate kernel
  std::vector<double> P_data;
  std::vector<c_int> P_indices;
  std::vector<c_int> P_indptr;
  CalculateKernel(&P_data, &P_indices, &P_indptr);

  // 计算仿射约束条件
  // A_data: 约束矩阵的非零元素值
  // A_indices: 非零元素的行索引
  // A_indptr: 列指针数组（CSC格式）
  // lower_bounds: 约束下界
  // upper_bounds: 约束上界
  // Calculate affine constraints
  std::vector<double> A_data;
  std::vector<c_int> A_indices;
  std::vector<c_int> A_indptr;
  std::vector<double> lower_bounds;
  std::vector<double> upper_bounds;
  CalculateAffineConstraint(&A_data, &A_indices, &A_indptr, &lower_bounds,
                            &upper_bounds);

  // Calculate offset
  // 计算线性项（偏移量）
  // q: 线性项系数向量
  std::vector<double> q;
  CalculateOffset(&q);

  // Set primal warm start
  // 设置原始变量的热启动值
  // primal_warm_start: 热启动向量，使用参考点坐标作为初始值
  std::vector<double> primal_warm_start;
  SetPrimalWarmStart(&primal_warm_start);

  // 分配OSQP求解器所需的数据结构内存
  OSQPData *data = reinterpret_cast<OSQPData *>(c_malloc(sizeof(OSQPData)));
  OSQPSettings *settings =
      reinterpret_cast<OSQPSettings *>(c_malloc(sizeof(OSQPSettings)));

  // Define Solver settings
  // 配置OSQP求解器参数
  osqp_set_default_settings(settings); // 设置默认参数
  settings->max_iter = max_iter_;      // 最大迭代次数
  // settings->time_limit = time_limit_;  // 时间限制
  settings->verbose = verbose_;        // 是否输出详细信息
  settings->scaled_termination = scaled_termination_; // 是否使用缩放终止条件
  settings->warm_start = warm_start_;                 // 是否启用热启动

  OSQPWorkspace *work = nullptr; // OSQP工作空间指针
  // 调用OSQP进行优化求解
  bool res = OptimizeWithOsqp(num_of_variables_, lower_bounds.size(), &P_data,
                              &P_indices, &P_indptr, &A_data, &A_indices,
                              &A_indptr, &lower_bounds, &upper_bounds, &q,
                              &primal_warm_start, data, &work, settings);
  // 检查优化结果是否成功
  if (res == false || work == nullptr || work->solution == nullptr) {
    AERROR << "Failed to find solution.";
    // Cleanup
    // 清理内存：释放OSQP工作空间和相关数据结构
    osqp_cleanup(work);
    c_free(data->A);
    c_free(data->P);
    c_free(data);
    c_free(settings);

    return false;
  }

  // Extract primal results
  // 提取优化结果：将解向量转换为x,y坐标序列
  x_.resize(num_of_points_); // 存储优化后的x坐标
  y_.resize(num_of_points_); // 存储优化后的y坐标
  for (int i = 0; i < num_of_points_; ++i) {
    int index = i * 2; // 每个点对应解向量中的两个连续元素
    x_.at(i) = work->solution->x[index];     // 提取x坐标
    y_.at(i) = work->solution->x[index + 1]; // 提取y坐标
  }

  // Cleanup
  // 最终清理：释放所有分配的内存
  osqp_cleanup(work);
  c_free(data->A);
  c_free(data->P);
  c_free(data);
  c_free(settings);

  return true; // 返回优化成功标志
}

void FemPosDeviationOsqpInterface::CalculateKernel(
    std::vector<double> *P_data, std::vector<c_int> *P_indices,
    std::vector<c_int> *P_indptr) {
  // 检查优化变量数量是否足够（至少需要4个变量，对应2个点）
  CHECK_GT(num_of_variables_, 4);

  // Three quadratic penalties are involved:
  // 1. Penalty x on distance between middle point and point by finite element
  // estimate;
  // 2. Penalty y on path length;
  // 3. Penalty z on difference between points and reference points

  // General formulation of P matrix is as below(with 6 points as an example):
  // I is a two by two identity matrix, X, Y, Z represents x * I, y * I, z * I
  // 0 is a two by two zero matrix
  // |X+Y+Z, -2X-Y,   X,       0,       0,       0    |
  // |0,     5X+2Y+Z, -4X-Y,   X,       0,       0    |
  // |0,     0,       6X+2Y+Z, -4X-Y,   X,       0    |
  // |0,     0,       0,       6X+2Y+Z, -4X-Y,   X    |
  // |0,     0,       0,       0,       5X+2Y+Z, -2X-Y|
  // |0,     0,       0,       0,       0,       X+Y+Z|

  // 优化目标函数包含三个二次惩罚项：
  // 1. 有限元位置偏差惩罚（x）：控制中间点与有限元估计点之间的距离
  // 2. 路径长度惩罚（y）：控制路径的总长度
  // 3. 参考点偏差惩罚（z）：控制优化点与参考点之间的偏差

  // P矩阵的一般形式（以6个点为例）：
  // I是2x2单位矩阵，X、Y、Z分别表示x*I、y*I、z*I
  // 0是2x2零矩阵
  // |X+Y+Z, -2X-Y,   X,       0,       0,       0    |
  // |0,     5X+2Y+Z, -4X-Y,   X,       0,       0    |
  // |0,     0,       6X+2Y+Z, -4X-Y,   X,       0    |
  // |0,     0,       0,       6X+2Y+Z, -4X-Y,   X    |
  // |0,     0,       0,       0,       5X+2Y+Z, -2X-Y|
  // |0,     0,       0,       0,       0,       X+Y+Z|

  // 由于P矩阵是对称的，只需要填充上三角部分
  // 使用列优先的存储方式，每列存储该列的非零元素
  // Only upper triangle needs to be filled
  std::vector<std::vector<std::pair<c_int, double>>> columns;
  columns.resize(num_of_variables_);
  int col_num = 0; // 列计数器
  // 处理前2列（对应第一个点的x和y坐标）
  for (int col = 0; col < 2; ++col) {
    // 对角线元素：X+Y+Z = weight_fem_pos_deviation_ + weight_path_length_ +
    // weight_ref_deviation_
    columns[col].emplace_back(col, weight_fem_pos_deviation_ +
                                       weight_path_length_ +
                                       weight_ref_deviation_);
    ++col_num;
  }
  // 处理第3-4列（对应第二个点的x和y坐标）
  for (int col = 2; col < 4; ++col) {
    // 与前一列的非对角线元素：-2X-Y
    columns[col].emplace_back(col - 2, -2.0 * weight_fem_pos_deviation_ -
                                           weight_path_length_);
    // 对角线元素：5X+2Y+Z
    columns[col].emplace_back(col, 5.0 * weight_fem_pos_deviation_ +
                                       2.0 * weight_path_length_ +
                                       weight_ref_deviation_);
    ++col_num;
  }

  int second_point_from_last_index = num_of_points_ - 2;
  for (int point_index = 2; point_index < second_point_from_last_index;
       ++point_index) {
    int col_index = point_index * 2;
    for (int col = 0; col < 2; ++col) {
      col_index += col;
      columns[col_index].emplace_back(col_index - 4, weight_fem_pos_deviation_);
      columns[col_index].emplace_back(col_index - 2,
                                      -4.0 * weight_fem_pos_deviation_ -
                                          weight_path_length_);
      columns[col_index].emplace_back(
          col_index, 6.0 * weight_fem_pos_deviation_ +
                         2.0 * weight_path_length_ + weight_ref_deviation_);
      ++col_num;
    }
  }

  int second_point_col_from_last_col = num_of_variables_ - 4;
  int last_point_col_from_last_col = num_of_variables_ - 2;
  for (int col = second_point_col_from_last_col;
       col < last_point_col_from_last_col; ++col) {
    columns[col].emplace_back(col - 4, weight_fem_pos_deviation_);
    columns[col].emplace_back(col - 2, -4.0 * weight_fem_pos_deviation_ -
                                           weight_path_length_);
    columns[col].emplace_back(col, 5.0 * weight_fem_pos_deviation_ +
                                       2.0 * weight_path_length_ +
                                       weight_ref_deviation_);
    ++col_num;
  }

  for (int col = last_point_col_from_last_col; col < num_of_variables_; ++col) {
    columns[col].emplace_back(col - 4, weight_fem_pos_deviation_);
    columns[col].emplace_back(col - 2, -2.0 * weight_fem_pos_deviation_ -
                                           weight_path_length_);
    columns[col].emplace_back(col, weight_fem_pos_deviation_ +
                                       weight_path_length_ +
                                       weight_ref_deviation_);
    ++col_num;
  }

  CHECK_EQ(col_num, num_of_variables_);

  int ind_p = 0;
  for (int i = 0; i < col_num; ++i) {
    P_indptr->push_back(ind_p);
    for (const auto &row_data_pair : columns[i]) {
      // Rescale by 2.0 as the quadratic term in osqp default qp problem setup
      // is set as (1/2) * x' * P * x
      P_data->push_back(row_data_pair.second * 2.0);
      P_indices->push_back(row_data_pair.first);
      ++ind_p;
    }
  }
  P_indptr->push_back(ind_p);
}

void FemPosDeviationOsqpInterface::CalculateOffset(std::vector<double> *q) {
  for (int i = 0; i < num_of_points_; ++i) {
    const auto &ref_point_xy = ref_points_[i];
    q->push_back(-2.0 * weight_ref_deviation_ * ref_point_xy.first);
    q->push_back(-2.0 * weight_ref_deviation_ * ref_point_xy.second);
  }
}

void FemPosDeviationOsqpInterface::CalculateAffineConstraint(
    std::vector<double> *A_data, std::vector<c_int> *A_indices,
    std::vector<c_int> *A_indptr, std::vector<double> *lower_bounds,
    std::vector<double> *upper_bounds) {
  int ind_A = 0;
  for (int i = 0; i < num_of_variables_; ++i) {
    A_data->push_back(1.0);
    A_indices->push_back(i);
    A_indptr->push_back(ind_A);
    ++ind_A;
  }
  A_indptr->push_back(ind_A);

  for (int i = 0; i < num_of_points_; ++i) {
    const auto &ref_point_xy = ref_points_[i];
    upper_bounds->push_back(ref_point_xy.first + bounds_around_refs_[i]);
    upper_bounds->push_back(ref_point_xy.second + bounds_around_refs_[i]);
    lower_bounds->push_back(ref_point_xy.first - bounds_around_refs_[i]);
    lower_bounds->push_back(ref_point_xy.second - bounds_around_refs_[i]);
  }
}

void FemPosDeviationOsqpInterface::SetPrimalWarmStart(
    std::vector<double> *primal_warm_start) {
  CHECK_EQ(ref_points_.size(), static_cast<size_t>(num_of_points_));
  for (const auto &ref_point_xy : ref_points_) {
    primal_warm_start->push_back(ref_point_xy.first);
    primal_warm_start->push_back(ref_point_xy.second);
  }
}

bool FemPosDeviationOsqpInterface::OptimizeWithOsqp(
    const size_t kernel_dim, const size_t num_affine_constraint,
    std::vector<double> *P_data, std::vector<c_int> *P_indices,
    std::vector<c_int> *P_indptr, std::vector<double> *A_data,
    std::vector<c_int> *A_indices, std::vector<c_int> *A_indptr,
    std::vector<double> *lower_bounds, std::vector<double> *upper_bounds,
    std::vector<double> *q, std::vector<double> *primal_warm_start,
    OSQPData *data, OSQPWorkspace **work, OSQPSettings *settings) {
  CHECK_EQ(lower_bounds->size(), upper_bounds->size());

  data->n = kernel_dim;
  data->m = num_affine_constraint;
  data->P = csc_matrix(data->n, data->n, P_data->size(), P_data->data(),
                       P_indices->data(), P_indptr->data());
  data->q = q->data();
  data->A = csc_matrix(data->m, data->n, A_data->size(), A_data->data(),
                       A_indices->data(), A_indptr->data());
  data->l = lower_bounds->data();
  data->u = upper_bounds->data();

  // *work = osqp_setup(data, settings);
  osqp_setup(work, data, settings);

  osqp_warm_start_x(*work, primal_warm_start->data());

  // Solve Problem
  osqp_solve(*work);

  auto status = (*work)->info->status_val;

  if (status < 0) {
    AERROR << "failed optimization status:\t" << (*work)->info->status;
    return false;
  }

  if (status != 1 && status != 2) {
    AERROR << "failed optimization status:\t" << (*work)->info->status;
    return false;
  }

  return true;
}

} // namespace algorithm
} // namespace planning