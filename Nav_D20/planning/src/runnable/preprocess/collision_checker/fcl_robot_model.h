#pragma once
#include "src/common/util/log.h"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <fcl/fcl.h>
#include <fcl/geometry/bvh/BVH_internal.h>
#include <fcl/geometry/bvh/BVH_model.h>
#include <fcl/geometry/geometric_shape_to_BVH_model.h>
#include <fcl/geometry/shape/box.h>
#include <fcl/geometry/shape/cylinder.h>
#include <fcl/geometry/shape/sphere.h>
#include <fcl/geometry/shape/triangle_p.h>
#include <string>
#include <urdf/model.h>

using fcl::BVHModel;
using fcl::CollisionObjectd;
using fcl::CollisionRequestd;
using fcl::CollisionResultd;
using fcl::Cylinderd;
using fcl::OBBRSSd;
using fcl::Sphered;
struct LinkCollision {
  std::vector<std::shared_ptr<CollisionObjectd>> objects;
  std::vector<Eigen::Isometry3d> local_transforms;
};

class FCLRobotModel {

public:
  FCLRobotModel(const std::string &urdf_dir, const std::string &urdf_name) {
    urdf::Model model;
    std::string urdf_path = urdf_dir + "/" + urdf_name;
    if (!model.initFile(urdf_path)) {
      throw std::runtime_error("Failed to load URDF file");
    }
    for (const auto &link_pair : model.links_) {
      const auto &link = link_pair.second;
      if (link->collision_array.empty())
        continue;

      LinkCollision link_collision;
      for (const auto &collision : link->collision_array) {
        if (!collision->geometry) {
          continue;
        }
        std::shared_ptr<CollisionObjectd> obj;
        Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
        tf.translation() << collision->origin.position.x,
            collision->origin.position.y, collision->origin.position.z;
        tf.linear() = Eigen::Quaterniond(collision->origin.rotation.w,
                                         collision->origin.rotation.x,
                                         collision->origin.rotation.y,
                                         collision->origin.rotation.z)
                          .toRotationMatrix();
        if (auto sphere =
                std::dynamic_pointer_cast<urdf::Sphere>(collision->geometry)) {
          auto fcl_sphere = std::make_shared<Sphered>(sphere->radius);
          obj = std::make_shared<CollisionObjectd>(fcl_sphere);
        } else if (auto cylinder = std::dynamic_pointer_cast<urdf::Cylinder>(
                       collision->geometry)) {
          auto fcl_cyl =
              std::make_shared<Cylinderd>(cylinder->radius, cylinder->length);
          obj = std::make_shared<CollisionObjectd>(fcl_cyl);
        } else if (auto box = std::dynamic_pointer_cast<urdf::Box>(
                       collision->geometry)) {
          auto fcl_box =
              std::make_shared<fcl::Boxd>(box->dim.x, box->dim.y, box->dim.z);
          obj = std::make_shared<CollisionObjectd>(fcl_box);
        } else if (auto mesh = std::dynamic_pointer_cast<urdf::Mesh>(
                       collision->geometry)) {
          std::string resolved_path = urdf_dir + "/" + mesh->filename; // TODO:

          static MeshCache mesh_cache;
          auto cached = mesh_cache.find(resolved_path);
          if (cached != mesh_cache.end()) {
            obj = std::make_shared<CollisionObjectd>(cached->second);
            AINFO << "Loaded mesh from cache for link: " << link->name;
          } else {
            auto fcl_mesh = loadMesh(resolved_path);
            if (fcl_mesh) {
              mesh_cache[resolved_path] = fcl_mesh;
              obj = std::make_shared<CollisionObjectd>(fcl_mesh);
              AINFO << "Loaded mesh from file for link: " << link->name;
            } else {
              AERROR << "Failed to load mesh: " << resolved_path;
              continue;
            }
          }
        } else {
          AERROR << "Unsupported collision geometry for link: " << link->name;
          continue;
        }
        link_collision.objects.emplace_back(obj);
        link_collision.local_transforms.emplace_back(tf);
      }
      links_[link->name] = link_collision;
    }
  }
  ~FCLRobotModel() {}

  void updateOctomap(const std::shared_ptr<const octomap::OcTree> &octree) {
    if (octree == nullptr)
      return;
    map_collison_object_.reset();
    std::shared_ptr<fcl::OcTreed> collision_geometry =
        std::make_shared<fcl::OcTreed>(octree);
    map_collison_object_ =
        std::make_shared<fcl::CollisionObjectd>(collision_geometry);

    // std::shared_ptr<fcl::CollisionGeometryd> collision_geometry(
    //     new fcl::OcTreed(octree));
    // map_collison_object_.reset(new
    // fcl::CollisionObjectd(collision_geometry));
    // map_collison_object_->setTransform(fcl::Quaterniond(1, 0, 0, 0),
    //                                        fcl::Vector3d(0, 0, -0.02));
  }

  void updateTransforms(const std::unordered_map<std::string, Eigen::Isometry3d>
                            &link_transforms) {
    for (auto &pair : links_) {
      const std::string &link_name = pair.first;
      auto &link_collison = pair.second;
      if (link_transforms.count(link_name)) {
        for (int i = 0; i < link_collison.objects.size(); ++i) {
          Eigen::Isometry3d global_tf = link_transforms.at(link_name) *
                                        link_collison.local_transforms.at(i);
          link_collison.objects.at(i)->setTransform(global_tf);
        }
      }
    }
  }

  bool checkSelfCollision(std::vector<Eigen::Vector3d> &contacts) {
    bool flag = false;
    for (auto it1 = links_.begin(); it1 != links_.end(); ++it1) {
      for (auto it2 = std::next(it1); it2 != links_.end(); ++it2) {
        if (it1->first == it2->first)
          continue;
        auto name_pair = std::minmax(it1->first, it2->first);
        if (ignore_pairs_.count(name_pair))
          continue;
        for (int i = 0; i < it1->second.objects.size(); ++i) {
          for (int j = 0; j < it2->second.objects.size(); ++j) {
            CollisionRequestd req;
            req.enable_contact = true;
            CollisionResultd res;
            fcl::collide(it1->second.objects[i].get(),
                         it2->second.objects[j].get(), req, res);
            if (res.isCollision()) {
              auto contact = res.getContact(0);
              //   publishMarker(marker_pub, contact);
              AINFO << "Collision between: " << it1->first << " & "
                    << it2->first << " , pos: " << contact.pos;
              contacts.emplace_back(contact.pos);
              flag = true;
            }
          }
        }
      }
    }
    return flag;
  }

  bool
  checkMapCollision(std::vector<Eigen::Vector3d> &contacts,
                    const Eigen::Vector3d &pose = Eigen::Vector3d::Zero()) {
    if (map_collison_object_ == nullptr) {
      return false;
    }
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() << pose.x(), pose.y(), 0.0;
    tf.linear() = Eigen::AngleAxisd(pose.z(), Eigen::Vector3d::UnitZ())
                      .toRotationMatrix();
    for (auto &pair : links_) {
      for (int i = 0; i < pair.second.objects.size(); ++i) {
        auto moved_link_object =
            std::make_shared<CollisionObjectd>(*pair.second.objects[i]);
        moved_link_object->setTransform(tf *
                                        pair.second.objects[i]->getTransform());

        CollisionRequestd req;
        req.enable_contact = true;
        CollisionResultd res;

        fcl::collide(moved_link_object.get(), map_collison_object_.get(), req,
                     res);
        if (res.isCollision()) {
          auto contact = res.getContact(0);
          AINFO << "Collision between: " << pair.first
                << " & map, pos: " << contact.pos;
          contacts.emplace_back(contact.pos);
          return true;
        }
      }
    }
    return false;
  }

  double getObstacleDistance(const Eigen::Vector3d &pose) {
    if (map_collison_object_ == nullptr) {
      return -1;
    }
    Eigen::Isometry3d tf = Eigen::Isometry3d::Identity();
    tf.translation() << pose.x(), pose.y(), 0.0;
    Eigen::Vector3d euler(pose.z(), 0, 0);
    Eigen::AngleAxisd Rx(euler(2), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd Ry(euler(1), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd Rz(euler(0), Eigen::Vector3d::UnitZ());
    Eigen::Quaterniond quaternion = Rz * Ry * Rx;
    tf.linear() = quaternion.toRotationMatrix();

    double min_dist = std::numeric_limits<double>::max();
    for (auto &pair : links_) {
      for (int i = 0; i < pair.second.objects.size(); ++i) {
        auto moved_link_object =
            std::make_shared<CollisionObjectd>(*pair.second.objects[i]);
        moved_link_object->setTransform(tf *
                                        pair.second.objects[i]->getTransform());

        fcl::DistanceRequestd req;
        req.enable_nearest_points = true;
        fcl::DistanceResultd res;

        fcl::distance(moved_link_object.get(), map_collison_object_.get(), req,
                      res);
        if (res.min_distance < min_dist) {
          min_dist = res.min_distance;
          AINFO << "Distance between: " << pair.first
                << " & map, dist: " << res.min_distance
                << ", nearest point on robot: " << res.nearest_points[0]
                << ", nearest point on map: " << res.nearest_points[1];
        }
      }
    }
    return min_dist;
  }

  const std::vector<std::string> getLinkNames() const {
    std::vector<std::string> names;
    for (const auto &pair : links_) {
      names.push_back(pair.first);
    }
    return names;
  }

private:
  std::shared_ptr<BVHModel<OBBRSSd>> loadMesh(const std::string &file_path) {
    auto model = std::make_shared<BVHModel<OBBRSSd>>();
    model->beginModel();

    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) {
      AERROR << "Failed to open mesh file: " << file_path;
      return nullptr;
    }

    file.seekg(80); // skip header
    uint32_t tri_count;
    file.read(reinterpret_cast<char *>(&tri_count), 4);

    for (uint32_t i = 0; i < tri_count; ++i) {
      file.seekg(12, std::ios::cur); // skip normal
      fcl::Vector3d v[3];
      for (int j = 0; j < 3; ++j) {
        float coords[3];
        file.read(reinterpret_cast<char *>(&coords[0]), sizeof(float));
        file.read(reinterpret_cast<char *>(&coords[1]), sizeof(float));
        file.read(reinterpret_cast<char *>(&coords[2]), sizeof(float));
        v[j] = fcl::Vector3d(coords[0], coords[1], coords[2]);
      }
      model->addTriangle(v[0], v[1], v[2]);
      file.seekg(2, std::ios::cur); // skip attribute byte count
    }

    model->endModel();
    model->computeLocalAABB();
    return model;
  }

  using MeshCache =
      std::unordered_map<std::string, std::shared_ptr<BVHModel<OBBRSSd>>>;
  std::set<std::pair<std::string, std::string>> ignore_pairs_;
  std::unordered_map<std::string, LinkCollision> links_;
  std::shared_ptr<CollisionObjectd> map_collison_object_;
};