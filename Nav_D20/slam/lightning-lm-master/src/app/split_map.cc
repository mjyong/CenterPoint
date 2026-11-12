//
// Created by Kilo Code on 2026-02-28.
// 将完整点云地图分块保存的工具程序
//

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "core/maps/tiled_map.h"
#include "common/point_def.h"
#include "io/file_io.h"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <string>
#include <filesystem>
#include <system_error>

// DEFINE_string(input_pcd, "", "/home/c37263/SaiLiSi/dog_robot/D20/lightning-lm-master/data/wuyuhu_1f/global.pcd");
// DEFINE_string(output_dir, "/home/c37263/SaiLiSi/dog_robot/D20/lightning-lm-master/data/wuyuhu_1f/map/", "输出分块地图的目录");
// DEFINE_double(chunk_size, 100.0, "分块大小（米）");
// DEFINE_double(voxel_size, 0.1, "块内体素滤波分辨率（米）");
// DEFINE_bool(overwrite, false, "是否覆盖已存在的输出目录");

using namespace lightning;
std::string input_pcd = "/home/c37263/SaiLiSi/dog_robot/D20/A2/lightning_arm_loc_new_map/data/test/merged/3d/global.pcd";
std::string output_dir = "/home/c37263/SaiLiSi/dog_robot/D20/A2/lightning_arm_loc_new_map/data/test/merged/3d";
double chunk_size = 100.0;
double voxel_size = 0.1;
bool overwrite = false;
int main(int argc, char** argv) {
    google::InitGoogleLogging(argv[0]);
    FLAGS_colorlogtostderr = true;
    FLAGS_stderrthreshold = google::INFO;

    google::ParseCommandLineFlags(&argc, &argv, true);

    // if (FLAGS_input_pcd.empty()) {
    //     LOG(ERROR) << "必须指定输入 PCD 文件路径 (--input_pcd)";
    //     return -1;
    // }

    // 检查输入文件是否存在
    if (!PathExists(input_pcd)) {
        LOG(ERROR) << "输入文件不存在: " << input_pcd;
        return -1;
    }

    // 创建输出目录
    if (!PathExists(output_dir)) {
        std::error_code ec;
        if (!std::filesystem::create_directories(output_dir, ec)) {
            LOG(ERROR) << "无法创建输出目录: " << output_dir << ", 错误: " << ec.message();
            return -1;
        }
    } else {
        if (!overwrite) {
            LOG(WARNING) << "输出目录已存在，如需覆盖请使用 --overwrite";
            // 继续执行，但可能已有文件存在
        }
    }

    // 加载完整点云
    CloudPtr full_cloud(new PointCloudType);
    if (pcl::io::loadPCDFile<PointType>(input_pcd, *full_cloud) == -1) {
        LOG(ERROR) << "无法加载 PCD 文件: " << input_pcd;
        return -1;
    }

    LOG(INFO) << "已加载点云，点数: " << full_cloud->size();

    // 配置 TiledMap 选项
    TiledMap::Options options;
    options.map_path_ = output_dir;
    options.chunk_size_ = chunk_size;
    options.inv_chunk_size_ = 1.0 / chunk_size;
    options.voxel_size_in_chunk_ = voxel_size;

    // 创建 TiledMap 对象
    TiledMap tiled_map(options);

    // 起始位姿设为原点（地图原点）
    SE3 start_pose = SE3();

    // 执行分块保存
    if (!tiled_map.ConvertFromFullPCD(full_cloud, start_pose, output_dir)) {
        LOG(ERROR) << "分块保存失败";
        return -1;
    }

    LOG(INFO) << "分块保存完成，地图已保存至: " << output_dir;
    LOG(INFO) << "索引文件: " << output_dir << "/index.txt";
    LOG(INFO) << "各区块文件: " << output_dir << "/*.pcd";

    return 0;
}