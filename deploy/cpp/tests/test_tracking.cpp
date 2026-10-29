// IMM filter step by step against Python; the tracker and predictor are checked on whole sequences in
// test_pipeline.cpp.
#include "dogperc/tracking/imm.hpp"
#include "golden.hpp"

namespace dogperc::test {

TEST(IMM, PredictUpdateRolloutMatchPython) {
  REQUIRE_GOLDEN();
  const Npz z = load("imm.npz");
  IMM imm({MotionModel::from_spec({"cv", 1.5, 1e-3}), MotionModel::from_spec({"ct", 1.5, 0.25})}, 0.95);
  StateVec x0;
  x0 << 2.0, -1.0, 3.0, 0.5, 0.0;
  StateVec d;
  d << 0.09, 0.09, 0.64, 0.64, 0.09;
  imm.initialize(x0, d.asDiagonal());
  const auto ops = member(z, "ops").to<int>();
  const auto dts = member(z, "dts").to<double>();
  const auto zs = member(z, "zs").to<double>();
  const auto ll = member(z, "loglik").to<double>();
  const NpyArray &X = member(z, "x"), &P = member(z, "P"), &MU = member(z, "mu");
  for (size_t k = 0; k < ops.size(); ++k) {
    double got_ll = kNaN;
    if (ops[k] == 0) {
      imm.predict(dts[k]);
    } else if (ops[k] == 1) {
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(4, 5);
      H.leftCols(4).setIdentity();
      Eigen::VectorXd zz(4);
      zz << zs[4 * k], zs[4 * k + 1], zs[4 * k + 2], zs[4 * k + 3];
      Eigen::Vector4d r(0.09, 0.09, 0.64, 0.64);
      got_ll = imm.update(zz, H, r.asDiagonal().toDenseMatrix());
    } else {
      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, 5);
      H.leftCols(2).setIdentity();
      Eigen::VectorXd zz(2);
      zz << zs[4 * k], zs[4 * k + 1];
      got_ll = imm.update(zz, H, Eigen::MatrixXd::Identity(2, 2) * 0.09);
    }
    for (int j = 0; j < 2; ++j) {
      for (int i = 0; i < 5; ++i) EXPECT_NEAR(imm.x()[j][i], X.at<double>((k * 2 + j) * 5 + i), 1e-10) << k;
      for (int i = 0; i < 25; ++i) EXPECT_NEAR(imm.P()[j](i / 5, i % 5), P.at<double>((k * 2 + j) * 25 + i), 1e-10) << k;
      EXPECT_NEAR(imm.mu()[j], MU.at<double>(k * 2 + j), 1e-12) << k;
    }
    if (ops[k] != 0) {
      EXPECT_NEAR(got_ll, ll[k], 1e-9) << k;
    }
  }
  const Rollout r = imm.rollout(3.0, 0.5);
  const NpyArray &M = member(z, "roll_means"), &C = member(z, "roll_covs");
  ASSERT_EQ(r.times.size(), 6u);
  for (int j = 0; j < 2; ++j) {
    for (int k = 0; k < 6; ++k) {
      for (int i = 0; i < 5; ++i) {
        EXPECT_NEAR(r.means[j][k][i], M.at<double>((j * 6 + k) * 5 + i), 1e-9);
      }
      for (int i = 0; i < 25; ++i) {
        EXPECT_NEAR(r.covs[j][k](i / 5, i % 5), C.at<double>((j * 6 + k) * 25 + i), 1e-9);
      }
    }
  }
}

}  // namespace dogperc::test
