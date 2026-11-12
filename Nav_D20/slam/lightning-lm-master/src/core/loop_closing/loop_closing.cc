//
// Created by xiang on 25-4-21.
//

#include "core/loop_closing/loop_closing.h"
#include "common/keyframe.h"
#include "common/loop_candidate.h"
#include "utils/pointcloud_utils.h"

#include <pcl/common/transforms.h>
#include <pcl/registration/ndt.h>

#include "core/opti_algo/algo_select.h"
#include "core/robust_kernel/cauchy.h"
#include "core/types/edge_se3.h"
#include "core/types/edge_se3_height_prior.h"
#include "core/types/edge_se3_prior.h"
#include "core/types/vertex_se3.h"
#include "io/yaml_io.h"
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/io.h> // 如果需要合并点云（可选）
namespace lightning {

LoopClosing::~LoopClosing() {
    if (options_.online_mode_) {
        kf_thread_.Quit();
    }
}
void pcl_visualize(CloudPtr cloud1, CloudPtr cloud2) {
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Two Point Clouds Viewer"));
    viewer->setBackgroundColor(0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color1(cloud1, 0, 255, 0);
    viewer->addPointCloud<PointType>(cloud1, single_color1, "cloud1");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud1");

    pcl::visualization::PointCloudColorHandlerCustom<PointType> single_color2(cloud2, 255, 0, 0);
    viewer->addPointCloud<PointType>(cloud2, single_color2, "cloud2");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud2");

    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();

    while (!viewer->wasStopped()) {
        viewer->spinOnce(100);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}


void LoopClosing::Init(const std::string yaml_path) {
    /// setup miao
    miao::OptimizerConfig config(miao::AlgorithmType::LEVENBERG_MARQUARDT,
                                 miao::LinearSolverType::LINEAR_SOLVER_SPARSE_EIGEN, false);
    config.incremental_mode_ = true;
    optimizer_ = miao::SetupOptimizer<6, 3>(config);

    info_motion_.setIdentity();
    info_motion_.block<3, 3>(0, 0) =
        Mat3d::Identity() * 1.0 / (options_.motion_trans_noise_ * options_.motion_trans_noise_);
    info_motion_.block<3, 3>(3, 3) =
        Mat3d::Identity() * 1.0 / (options_.motion_rot_noise_ * options_.motion_rot_noise_);

    info_loops_.setIdentity();
    info_loops_.block<3, 3>(0, 0) = Mat3d::Identity() * 1.0 / (options_.loop_trans_noise_ * options_.loop_trans_noise_);
    info_loops_.block<3, 3>(3, 3) = Mat3d::Identity() * 1.0 / (options_.loop_rot_noise_ * options_.loop_rot_noise_);

    // INS 先验约束信息矩阵初始化
    info_ins_.setIdentity();
    info_ins_.block<3, 3>(0, 0) = Mat3d::Identity() * 1.0 / (options_.ins_trans_noise_ * options_.ins_trans_noise_);
    info_ins_.block<3, 3>(3, 3) = Mat3d::Identity() * 1.0 / (options_.ins_rot_noise_ * options_.ins_rot_noise_);

    if (!yaml_path.empty()) {
        YAML_IO yaml(yaml_path);

        options_.loop_kf_gap_ = yaml.GetValue<int>("loop_closing", "loop_kf_gap");
        options_.min_id_interval_ = yaml.GetValue<int>("loop_closing", "min_id_interval");
        options_.closest_id_th_ = yaml.GetValue<int>("loop_closing", "closest_id_th");
        options_.max_range_ = yaml.GetValue<double>("loop_closing", "max_range");
        options_.ndt_score_th_ = yaml.GetValue<double>("loop_closing", "ndt_score_th");
        options_.with_height_ = yaml.GetValue<bool>("loop_closing", "with_height");
        
        // 读取 INS 先验约束配置
        // 使用 yaml-cpp 直接读取，因为 YAML_IO 没有默认值参数
        auto yaml_node = YAML::LoadFile(yaml_path);
        if (yaml_node["loop_closing"]["with_ins_prior"]) {
            options_.with_ins_prior_ = yaml_node["loop_closing"]["with_ins_prior"].as<bool>();
        }
        if (yaml_node["loop_closing"]["ins_trans_noise"]) {
            options_.ins_trans_noise_ = yaml_node["loop_closing"]["ins_trans_noise"].as<double>();
        }
        if (yaml_node["loop_closing"]["ins_rot_noise"]) {
            options_.ins_rot_noise_ = yaml_node["loop_closing"]["ins_rot_noise"].as<double>();
        }
        if (yaml_node["loop_closing"]["use_ins_covariance"]) {
            options_.use_ins_covariance_ = yaml_node["loop_closing"]["use_ins_covariance"].as<bool>();
        }
        if (yaml_node["loop_closing"]["ins_prior_position_only"]) {
            options_.ins_prior_position_only_ = yaml_node["loop_closing"]["ins_prior_position_only"].as<bool>();
        }
        if (yaml_node["loop_closing"]["ins_constraint_gap"]) {
            options_.ins_constraint_gap_ = yaml_node["loop_closing"]["ins_constraint_gap"].as<double>();
        }
        if (yaml_node["loop_closing"]["motion_trans_noise"]) {
            options_.motion_trans_noise_ = yaml_node["loop_closing"]["motion_trans_noise"].as<double>();
        }
        if (yaml_node["loop_closing"]["motion_rot_noise"]) {
            options_.motion_rot_noise_ = yaml_node["loop_closing"]["motion_rot_noise"].as<double>();
        }
        
        // 重新计算 INS 信息矩阵（因为噪声参数可能已从 YAML 更新）
        info_ins_.block<3, 3>(0, 0) = Mat3d::Identity() * 1.0 / (options_.ins_trans_noise_ * options_.ins_trans_noise_);
        info_ins_.block<3, 3>(3, 3) = Mat3d::Identity() * 1.0 / (options_.ins_rot_noise_ * options_.ins_rot_noise_);
        
        // 重新计算运动约束信息矩阵（因为噪声参数可能已从 YAML 更新）
        info_motion_.block<3, 3>(0, 0) = Mat3d::Identity() * 1.0 / (options_.motion_trans_noise_ * options_.motion_trans_noise_);
        info_motion_.block<3, 3>(3, 3) = Mat3d::Identity() * 1.0 / (options_.motion_rot_noise_ * options_.motion_rot_noise_);
    }

    if (options_.online_mode_) {
        LOG(INFO) << "loop closing module is running in online mode";
        kf_thread_.SetProcFunc([this](Keyframe::Ptr kf) { HandleKF(kf); });
        kf_thread_.SetName("handle loop closure");
        kf_thread_.Start();
    }
}

void LoopClosing::AddKF(Keyframe::Ptr kf) {
    if (options_.online_mode_) {
        kf_thread_.AddMessage(kf);
    } else {
        HandleKF(kf);
    }
}

void LoopClosing::HandleKF(Keyframe::Ptr kf) {
    if (kf == last_kf_) {
        return;
    }

    cur_kf_ = kf;
    all_keyframes_.emplace_back(kf);
    if(0) {
        // 检测回环候选
        DetectLoopCandidates();

        if (options_.verbose_) {
            LOG(INFO) << "lc: get kf " << cur_kf_->GetID() << " candi: " << candidates_.size();
        }

        // 计算回环位姿
        ComputeLoopCandidates();

        // 位姿图优化
        PoseOptimization();
    }
    last_kf_ = kf;
}

void LoopClosing::DetectLoopCandidates() {
    candidates_.clear();

    auto& kfs_mapping = all_keyframes_;
    Keyframe::Ptr check_first = nullptr;

    if (last_loop_kf_ == nullptr) {
        last_loop_kf_ = cur_kf_;
        return;
    }

    if (last_loop_kf_ && (cur_kf_->GetID() - last_loop_kf_->GetID()) <= options_.loop_kf_gap_) {
        LOG(INFO) << "skip because last loop kf: " << last_loop_kf_->GetID();
        return;
    }

    for (auto kf : kfs_mapping) {
        if (check_first != nullptr && abs(int(kf->GetID() - check_first->GetID())) <= options_.min_id_interval_) {
            // 同条轨迹内，跳过一定的ID区间
            continue;
        }

        if (abs(int(kf->GetID() - cur_kf_->GetID())) < options_.closest_id_th_) {
            /// 在同一条轨迹中，如果间隔太近，就不考虑回环
            break;
        }

        Vec3d dt = kf->GetLIOPose().translation() - cur_kf_->GetLIOPose().translation();
        double t2d = dt.head<2>().norm();  // x-y distance
        double range_th = options_.max_range_;

        if (t2d < range_th) {
            LoopCandidate c(kf->GetID(), cur_kf_->GetID());
            c.Tij_ = kf->GetLIOPose().inverse() * cur_kf_->GetLIOPose();

            candidates_.emplace_back(c);
            check_first = kf;
        }
    }

    if (!candidates_.empty()) {
        last_loop_kf_ = cur_kf_;
    }

    if (options_.verbose_ && !candidates_.empty()) {
        LOG(INFO) << "lc candi: " << candidates_.size();
    }
}

void LoopClosing::ComputeLoopCandidates() {
    if (candidates_.empty()) {
        return;
    }

    // 执行计算
    std::for_each(candidates_.begin(), candidates_.end(), [this](LoopCandidate& c) { ComputeForCandidate(c); });
    // 保存成功的候选
    std::vector<LoopCandidate> succ_candidates;
    for (const auto& lc : candidates_) {
        LOG(INFO) << "candi " << lc.idx1_ << ", " << lc.idx2_ << " s: " << lc.ndt_score_;
        if (lc.ndt_score_ > options_.ndt_score_th_) {
            succ_candidates.emplace_back(lc);
        }
    }

    if (options_.verbose_) {
        LOG(INFO) << "success: " << succ_candidates.size() << "/" << candidates_.size();
    }

    candidates_.swap(succ_candidates);
}

void LoopClosing::ComputeForCandidate(lightning::LoopCandidate& c) {
    LOG(INFO) << "aligning " << c.idx1_ << " with " << c.idx2_;
    const int submap_idx_range = 40;
    auto kf1 = all_keyframes_.at(c.idx1_), kf2 = all_keyframes_.at(c.idx2_);

    auto build_submap = [this](int given_id, bool build_in_world) -> CloudPtr {
        CloudPtr submap(new PointCloudType);
        for (int idx = -submap_idx_range; idx < submap_idx_range; idx += 4) {
            int id = idx + given_id;
            if (id < 0 || id > all_keyframes_.size()) {
                continue;
            }

            auto kf = all_keyframes_[id];
            CloudPtr cloud = kf->GetCloud();

            // RemoveGround(cloud, 0.1);

            if (cloud->empty()) {
                continue;
            }

            // 转到世界系下（使用优化位姿，即 INS 坐标系）
            SE3 Twb = kf->GetLIOPose();

            if (!build_in_world) {
                Twb = all_keyframes_.at(given_id)->GetLIOPose().inverse() * Twb;
            }

            CloudPtr cloud_trans(new PointCloudType);
            pcl::transformPointCloud(*cloud, *cloud_trans, Twb.matrix());

            *submap += *cloud_trans;
        }
        return submap;
    };

    auto submap_kf1 = build_submap(kf1->GetID(), true);

    CloudPtr submap_kf2 = kf2->GetCloud();

    if (submap_kf1->empty() || submap_kf2->empty()) {
        c.ndt_score_ = 0;
        return;
    }

    Mat4f Tw2 = kf2->GetLIOPose().matrix().cast<float>();

    /// 不同分辨率下的匹配
    CloudPtr output(new PointCloudType);
    std::vector<double> res{10.0, 5.0, 2.0, 1.0};

    CloudPtr rough_map1, rough_map2;

    for (auto& r : res) {
        pcl::NormalDistributionsTransform<PointType, PointType> ndt;
        ndt.setTransformationEpsilon(0.05);
        ndt.setStepSize(0.7);
        ndt.setMaximumIterations(40);

        ndt.setResolution(r);
        rough_map1 = VoxelGrid(submap_kf1, r * 0.1);
        rough_map2 = VoxelGrid(submap_kf2, r * 0.1);
        ndt.setInputTarget(rough_map1);
        ndt.setInputSource(rough_map2);

        ndt.align(*output, Tw2);
        Tw2 = ndt.getFinalTransformation();

        c.ndt_score_ = ndt.getTransformationProbability();
    } 

    Mat4d T = Tw2.cast<double>();
    Quatd q(T.block<3, 3>(0, 0));
    q.normalize();
    Vec3d t = T.block<3, 1>(0, 3);

    c.Tij_ = kf1->GetLIOPose().inverse() * SE3(q, t);
    // std::cout << "NDT score: " << c.ndt_score_ << std::endl;
    // std::cout << "c.Tij_: " << c.Tij_.matrix() << std::endl;
    //pcl_visualize(rough_map1, output);
    // *output = *output+*rough_map1;
    // pcl::io::savePCDFileBinaryCompressed(
    //     "./data/lc_" + std::to_string(c.idx1_) + "_" + std::to_string(c.idx2_) + "_out.pcd", *output);
    // pcl::io::savePCDFileBinaryCompressed(
    //     "./data/lc_" + std::to_string(c.idx1_) + "_" + std::to_string(c.idx2_) + "_tgt.pcd", *rough_map1);
}

void LoopClosing::PoseOptimization() {
    auto v = std::make_shared<miao::VertexSE3>();
    v->SetId(cur_kf_->GetID());
    v->SetEstimate(cur_kf_->GetLIOPose());

    optimizer_->AddVertex(v);
    kf_vert_.emplace_back(v);

    /// 上一个关键帧的运动约束
    /// TODO 3D激光最好是跟前面多个帧都有关联

    for (int i = 1; i < 3; i++) {
        int id = cur_kf_->GetID() - i;
        if (id >= 0) {
            auto last_kf = all_keyframes_[id];
            auto e = std::make_shared<miao::EdgeSE3>();
            e->SetVertex(0, optimizer_->GetVertex(last_kf->GetID()));
            e->SetVertex(1, v);
            // 运动约束：使用优化位姿（INS坐标系）计算运动
            SE3 motion = last_kf->GetLIOPose().inverse() * cur_kf_->GetLIOPose();
            e->SetMeasurement(motion);
            e->SetInformation(info_motion_);
            optimizer_->AddEdge(e);
        }
    }

    if (options_.with_height_) {
        /// 高度约束
        auto e = std::make_shared<miao::EdgeHeightPrior>();
        e->SetVertex(0, v);
        e->SetMeasurement(0);
        e->SetInformation(Mat1d::Identity() * 1.0 / (options_.height_noise_ * options_.height_noise_));
        optimizer_->AddEdge(e);
    }

    /// 回环的约束
    for (auto& c : candidates_) {
        auto e = std::make_shared<miao::EdgeSE3>();
        e->SetVertex(0, optimizer_->GetVertex(c.idx1_));
        e->SetVertex(1, optimizer_->GetVertex(c.idx2_));
        
        // 回环约束 Tij_ 现在是在 INS 坐标系中计算的
        e->SetMeasurement(c.Tij_);
        e->SetInformation(info_loops_);

        auto rk = std::make_shared<miao::RobustKernelCauchy>();
        rk->SetDelta(options_.rk_loop_th_);
        e->SetRobustKernel(rk);

        optimizer_->AddEdge(e);
        edge_loops_.emplace_back(e);
    }

    if (optimizer_->GetEdges().empty()) {
        return;
    }

    if (candidates_.empty()) {
        return;
    }

    optimizer_->InitializeOptimization();
    optimizer_->SetVerbose(false);

    optimizer_->Optimize(20);

    /// remove outliers
    int cnt_outliers = 0;
    for (auto& e : edge_loops_) {
        if (e->GetRobustKernel() == nullptr) {
            continue;
        }

        if (e->Chi2() > e->GetRobustKernel()->Delta()) {
            e->SetLevel(1);
            cnt_outliers++;
        } else {
            e->SetRobustKernel(nullptr);
        }
    }

    if (options_.verbose_) {
        LOG(INFO) << "loop outliers: " << cnt_outliers << "/" << edge_loops_.size();
    }

    /// get results
    for (auto& vert : kf_vert_) {
        SE3 pose = vert->Estimate();
        all_keyframes_[vert->GetId()]->SetLIOPose(pose);
    }

    if (loop_cb_) {
        loop_cb_();
    }

    LOG(INFO) << "optimize finished, loops: " << edge_loops_.size();

    // LOG(INFO) << "lc: cur kf " << cur_kf_->GetID() << ", opt: " << cur_kf_->GetLIOPose().translation().transpose()
    //           << ", lio: " << cur_kf_->GetLIOPose().translation().transpose();
}

void LoopClosing::GlobalOptimizationWithINS() {
    LOG(INFO) << "Starting global optimization with INS constraints";
    
    // 启用INS先验约束
    bool original_ins_prior = options_.with_ins_prior_;
    options_.with_ins_prior_ = true;
    
    // 检查外参是否有效，如果无效则尝试从第一个有INS位姿的关键帧计算
    if (T_ins_lio_.matrix().isZero(1e-6)) {
        for (auto& kf : all_keyframes_) {
            if (kf->HasInsPose()) {
                SE3 lio_pose = kf->GetLIOPose();  // LIO坐标系下的优化位姿
                SE3 ins_pose = kf->GetInsPose();
                T_ins_lio_ = ins_pose * lio_pose.inverse();
                break;
            }
        }
    }
    if (T_ins_lio_.matrix().isZero(1e-6)) {
        LOG(WARNING) << "T_ins_lio not available, skipping global optimization with INS";
        options_.with_ins_prior_ = original_ins_prior;
        return;
    }
    
    // 清除优化器（保留配置）
    optimizer_->Clear();
    kf_vert_.clear();
    edge_loops_.clear();
    
    // 添加所有关键帧顶点，初始估计转换为INS坐标系
    for (auto& kf : all_keyframes_) {
        auto v = std::make_shared<miao::VertexSE3>();
        v->SetId(kf->GetID());
        SE3 lio_pose = kf->GetLIOPose();
        SE3 ins_pose_initial = T_ins_lio_ * lio_pose;
        LOG(INFO) << "Computed pose from KF " << kf->GetID() << ": " << ins_pose_initial.translation().transpose();
        LOG(INFO) << "Computed ins from KF " << kf->GetID() << ": " << kf->GetInsPose().translation().transpose();
        v->SetEstimate(ins_pose_initial);
        optimizer_->AddVertex(v);
        kf_vert_.emplace_back(v);
    }
    
    // 添加运动约束（相邻关键帧之间的约束），在INS坐标系下相对运动与LIO坐标系下相同
    for (size_t i = 1; i < all_keyframes_.size(); ++i) {
        auto kf_prev = all_keyframes_[i-1];
        auto kf_curr = all_keyframes_[i];
        auto e = std::make_shared<miao::EdgeSE3>();
        e->SetVertex(0, optimizer_->GetVertex(kf_prev->GetID()));
        e->SetVertex(1, optimizer_->GetVertex(kf_curr->GetID()));
        SE3 motion = kf_prev->GetLIOPose().inverse() * kf_curr->GetLIOPose();  // 相对运动在LIO坐标系下
        e->SetMeasurement(motion);  // 注意：相对运动在INS坐标系下与LIO坐标系下相同（因为外参抵消）
        e->SetInformation(info_motion_);
        optimizer_->AddEdge(e);
    }
    
    // 添加回环约束（从已有的candidates_中获取），同样相对运动不变
    // 注意：candidates_可能只包含最近的回环候选，我们需要存储所有历史回环
    // 为了简化，我们暂时只使用当前candidates_（可能为空）
    // 更好的做法是存储所有成功的回环约束，但这里我们假设回环已经通过在线优化处理过了
    // 因此我们跳过回环约束，只添加INS约束
    
    // 添加INS先验约束
    // 按里程间隔抽稀：每累计 ins_constraint_gap_ 米里程才融合一个INS点（0 表示不抽稀，每个关键帧都加）
    // 首个和最后一个带INS位姿的关键帧始终加入，用于锚定轨迹两端
    std::vector<Keyframe::Ptr> ins_kfs;
    int ins_kf_total = 0;
    {
        double accum_dist = 0.0;               // 距上一个被选中关键帧的累计里程
        Vec3d prev_pos = Vec3d::Zero();        // 上一关键帧位置
        bool has_prev = false;
        Keyframe::Ptr pending_tail = nullptr;  // 暂存未选中的带INS关键帧，作为尾部锚点候选
        for (auto& kf : all_keyframes_) {
            if (has_prev) {
                accum_dist += (kf->GetLIOPose().translation() - prev_pos).norm();
            }
            prev_pos = kf->GetLIOPose().translation();
            has_prev = true;

            if (!kf->HasInsPose()) {
                continue;
            }
            ins_kf_total++;
            if (ins_kfs.empty() || accum_dist >= options_.ins_constraint_gap_) {
                ins_kfs.emplace_back(kf);
                accum_dist = 0.0;
                pending_tail = nullptr;
            } else {
                pending_tail = kf;
            }
        }
        if (pending_tail != nullptr) {
            ins_kfs.emplace_back(pending_tail);  // 保证轨迹末端也有INS锚点
        }
    }
    LOG(INFO) << "INS prior constraint subsampling: gap = " << options_.ins_constraint_gap_
              << " m, selected " << ins_kfs.size() << "/" << ins_kf_total << " keyframes with INS pose";

    int ins_constraint_count = 0;
    for (auto& kf : ins_kfs) {
        auto e = std::make_shared<miao::EdgeSE3Prior>();
        e->SetVertex(0, optimizer_->GetVertex(kf->GetID()));
        e->SetMeasurement(SE3(kf->GetInsPose()));
        Mat6d info_ins_current = info_ins_;
        if (options_.use_ins_covariance_) {
            Eigen::Vector3d pos_cov = kf->GetInsPositionCovariance();
            Eigen::Vector3d ori_cov = kf->GetInsOrientationCovariance();
            if (pos_cov.norm() > 1e-6 && ori_cov.norm() > 1e-6) {
                Mat6d info_from_cov = Mat6d::Zero();
                info_from_cov(0,0) = 1.0 / (pos_cov.x() * pos_cov.x());
                info_from_cov(1,1) = 1.0 / (pos_cov.y() * pos_cov.y());
                info_from_cov(2,2) = 1.0 / (pos_cov.z() * pos_cov.z());
                info_from_cov(3,3) = 1.0 / (ori_cov.x() * ori_cov.x());
                info_from_cov(4,4) = 1.0 / (ori_cov.y() * ori_cov.y());
                info_from_cov(5,5) = 1.0 / (ori_cov.z() * ori_cov.z());
                info_ins_current = info_from_cov;
            }
        }
        if (options_.ins_prior_position_only_) {
            info_ins_current.block<3,3>(3,3) *= 1e-9;
        }
        // 降低INS约束的权重，避免数值问题
        // info_ins_current *= 0.1;
        e->SetInformation(info_ins_current);

        auto rk = std::make_shared<miao::RobustKernelCauchy>();
        rk->SetDelta(1.0);
        e->SetRobustKernel(rk);
        optimizer_->AddEdge(e);
        ins_constraint_count++;
    }
    if(ins_constraint_count==0) {
        LOG(WARNING) << "No INS prior constraints added, skipping global optimization with INS";
        options_.with_ins_prior_ = original_ins_prior;
        return;
    }
    LOG(INFO) << "Added " << ins_constraint_count << " INS prior constraints";
    
    if (optimizer_->GetEdges().empty()) {
        LOG(WARNING) << "No edges in optimizer, skipping optimization";
        options_.with_ins_prior_ = original_ins_prior;
        return;
    }
    
    optimizer_->InitializeOptimization();
    optimizer_->SetVerbose(true);
    optimizer_->Optimize(10);
    
    // 获取优化结果（INS坐标系下的位姿）并更新关键帧位姿（转换为LIO坐标系）
    for (auto& vert : kf_vert_) {
        SE3 ins_pose_opt = vert->Estimate();
        //SE3 lio_pose = kf->GetLIOPose();
        //SE3 lio_pose_opt = T_ins_lio_.inverse() * ins_pose_opt;
        all_keyframes_[vert->GetId()]->SetOptPose(ins_pose_opt);
    }
    
    LOG(INFO) << "Global optimization with INS completed (optimized in INS frame)";
    
    // 统计优化后的位姿与INS位姿的差异
    double max_trans_err = 0.0;
    double min_trans_err = 1e9;
    double avg_trans_err = 0.0;
    double max_rot_err = 0.0;
    double min_rot_err = 1e9;
    double avg_rot_err = 0.0;
    int count = 0;
    for (auto& kf : all_keyframes_) {
        if (kf->HasInsPose()) {
            SE3 opt_pose = kf->GetOptPose();  // 优化后的位姿（INS坐标系下）
            SE3 ins_pose = kf->GetInsPose();
            SE3 diff = ins_pose.inverse() * opt_pose;
            double trans_err = diff.translation().norm();
            double rot_err = diff.so3().log().norm();
            max_trans_err = std::max(max_trans_err, trans_err);
            min_trans_err = std::min(min_trans_err, trans_err);
            max_rot_err = std::max(max_rot_err, rot_err);
            min_rot_err = std::min(min_rot_err, rot_err);
            avg_trans_err += trans_err;
            avg_rot_err += rot_err;
            count++;
        }
    }
    if (count > 0) {
        avg_trans_err /= count;
        avg_rot_err /= count;
        LOG(INFO) << "INS vs optimized pose error statistics (count=" << count << "):";
        LOG(INFO) << "  Translation error: max=" << max_trans_err << " m, min=" << min_trans_err << " m, avg=" << avg_trans_err << " m";
        LOG(INFO) << "  Rotation error: max=" << max_rot_err << " rad, min=" << min_rot_err << " rad, avg=" << avg_rot_err << " rad";
    }
    
    // 恢复原始设置
    options_.with_ins_prior_ = original_ins_prior;
}

}  // namespace lightning
