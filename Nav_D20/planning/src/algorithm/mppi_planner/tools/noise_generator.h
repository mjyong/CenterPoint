// Ported from mppi_controller_ros (Apache-2.0, Samsung Research America /
// @artofnothingness Alexey Budyakov). Parameters are passed directly instead
// of through a Parameters struct.

#pragma once

#include <Eigen/Dense>

#include <condition_variable>
#include <mutex>
#include <random>
#include <thread>

#include "src/algorithm/mppi_planner/models/control_sequence.h"
#include "src/algorithm/mppi_planner/models/optimizer_settings.h"
#include "src/algorithm/mppi_planner/models/state.h"

namespace mppi {

/**
 * @class mppi::NoiseGenerator
 * @brief Generates noise trajectories from optimal trajectory
 */
class NoiseGenerator {
public:
  NoiseGenerator() = default;
  ~NoiseGenerator() { shutdown(); }

  /**
   * @brief Initialize noise generator with settings and model types
   */
  void initialize(
      const mppi::models::OptimizerSettings &settings, bool is_holonomic,
      bool regenerate_noises) {
    settings_ = settings;
    is_holonomic_ = is_holonomic;
    active_ = true;
    regenerate_noises_ = regenerate_noises;

    ndistribution_vx_ = std::normal_distribution<float>(0.0f, settings_.sampling_std.vx);
    ndistribution_vy_ = std::normal_distribution<float>(0.0f, settings_.sampling_std.vy);
    ndistribution_wz_ = std::normal_distribution<float>(0.0f, settings_.sampling_std.wz);

    if (regenerate_noises_) {
      noise_thread_ = std::thread(std::bind(&NoiseGenerator::noiseThread, this));
    } else {
      generateNoisedControls();
    }
  }

  /**
   * @brief Shutdown noise generator thread
   */
  void shutdown() {
    active_ = false;
    {
      std::unique_lock<std::mutex> guard(noise_lock_);
      ready_ = true;
    }
    noise_cond_.notify_all();
    if (noise_thread_.joinable()) {
      noise_thread_.join();
    }
  }

  /**
   * @brief Signal to the noise thread the controller is ready to generate a
   * new noised control for the next iteration
   */
  void generateNextNoises() {
    {
      std::unique_lock<std::mutex> guard(noise_lock_);
      ready_ = true;
    }
    noise_cond_.notify_all();
  }

  /**
   * @brief set noised control_sequence to state controls
   */
  void setNoisedControls(
      models::State &state, const models::ControlSequence &control_sequence) {
    std::unique_lock<std::mutex> guard(noise_lock_);

    state.cvx = noises_vx_.rowwise() + control_sequence.vx.transpose();
    state.cvy = noises_vy_.rowwise() + control_sequence.vy.transpose();
    state.cwz = noises_wz_.rowwise() + control_sequence.wz.transpose();
  }

  /**
   * @brief Reset noise generator with settings and model types
   */
  void reset(const mppi::models::OptimizerSettings &settings, bool is_holonomic) {
    settings_ = settings;
    is_holonomic_ = is_holonomic;

    // Recompute the noises on reset, initialization, and fallback
    {
      std::unique_lock<std::mutex> guard(noise_lock_);
      noises_vx_.setZero(settings_.batch_size, settings_.time_steps);
      noises_vy_.setZero(settings_.batch_size, settings_.time_steps);
      noises_wz_.setZero(settings_.batch_size, settings_.time_steps);
      ready_ = true;
    }

    if (regenerate_noises_) {
      noise_cond_.notify_all();
    } else {
      generateNoisedControls();
    }
  }

protected:
  /**
   * @brief Thread to execute noise generation process
   */
  void noiseThread() {
    do {
      std::unique_lock<std::mutex> guard(noise_lock_);
      noise_cond_.wait(guard, [this]() { return ready_; });
      ready_ = false;
      if (!active_) {
        break;
      }
      generateNoisedControls();
    } while (active_);
  }

  /**
   * @brief Generate random controls by gaussian noise with mean in
   * control_sequence_
   */
  void generateNoisedControls() {
    auto &s = settings_;
    noises_vx_ = Eigen::ArrayXXf::NullaryExpr(
        s.batch_size, s.time_steps,
        [&]() { return ndistribution_vx_(generator_); });
    noises_wz_ = Eigen::ArrayXXf::NullaryExpr(
        s.batch_size, s.time_steps,
        [&]() { return ndistribution_wz_(generator_); });
    if (is_holonomic_) {
      noises_vy_ = Eigen::ArrayXXf::NullaryExpr(
          s.batch_size, s.time_steps,
          [&]() { return ndistribution_vy_(generator_); });
    }
  }

  Eigen::ArrayXXf noises_vx_;
  Eigen::ArrayXXf noises_vy_;
  Eigen::ArrayXXf noises_wz_;

  std::default_random_engine generator_;
  std::normal_distribution<float> ndistribution_vx_;
  std::normal_distribution<float> ndistribution_wz_;
  std::normal_distribution<float> ndistribution_vy_;

  mppi::models::OptimizerSettings settings_;
  bool is_holonomic_{false};

  std::thread noise_thread_;
  std::condition_variable noise_cond_;
  std::mutex noise_lock_;
  bool active_{false}, ready_{false}, regenerate_noises_{false};
};

} // namespace mppi
