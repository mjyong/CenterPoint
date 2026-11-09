#include "src/common/path2d.h"

namespace planning {
Path2d::Path2d(const std::vector<Point2d> &points) {
  std::vector<double> headings, s_vec, kappas, dkappas;
  if (!computePathProfile(points, &headings, &s_vec, &kappas, &dkappas)) {
    return;
  }
  points_.clear();
  poses_.clear();
  for (size_t i = 0; i < headings.size(); i++) {
    PathPoint2d point(points[i], s_vec[i], headings[i], kappas[i], dkappas[i]);
    points_.emplace_back(point);
    poses_.emplace_back(points[i], headings[i]);
  }
  s_vec_ = std::move(s_vec);
}

Path2d::Path2d(const std::vector<Pose2d> &poses) {
  std::vector<Point2d> points;
  for (const auto &pose : poses) {
    points.emplace_back(pose.position);
  }
  std::vector<double> headings, s_vec, kappas, dkappas;
  if (!computePathProfile(points, &headings, &s_vec, &kappas, &dkappas)) {
    return;
  }
  points_.clear();
  for (size_t i = 0; i < headings.size(); i++) {
    PathPoint2d point(points[i], s_vec[i], poses_[i].theta, kappas[i],
                      dkappas[i]);
    points_.emplace_back(point);
  }
  s_vec_ = std::move(s_vec);
  poses_ = poses;
}

bool Path2d::computePathProfile(const std::vector<Point2d> &xy_points,
                                std::vector<double> *headings,
                                std::vector<double> *s_vec,
                                std::vector<double> *kappas,
                                std::vector<double> *dkappas) {
  headings->clear();
  kappas->clear();
  dkappas->clear();
  ACHECK(xy_points.size() > 1);
  std::vector<double> dxs;
  std::vector<double> dys;
  std::vector<double> y_over_s_first_derivatives;
  std::vector<double> x_over_s_first_derivatives;
  std::vector<double> y_over_s_second_derivatives;
  std::vector<double> x_over_s_second_derivatives;

  std::size_t points_size = xy_points.size();
  for (std::size_t i = 0; i < points_size; ++i) {
    double x_delta = 0.0;
    double y_delta = 0.0;
    if (i == 0) {
      x_delta = (xy_points[i + 1].x() - xy_points[i].x());
      y_delta = (xy_points[i + 1].y() - xy_points[i].y());
    } else if (i == points_size - 1) {
      x_delta = (xy_points[i].x() - xy_points[i - 1].x());
      y_delta = (xy_points[i].y() - xy_points[i - 1].y());
    } else {
      x_delta = 0.5 * (xy_points[i + 1].x() - xy_points[i - 1].x());
      y_delta = 0.5 * (xy_points[i + 1].y() - xy_points[i - 1].y());
    }
    dxs.push_back(x_delta);
    dys.push_back(y_delta);
  }

  for (std::size_t i = 0; i < points_size; ++i) {
    headings->push_back(std::atan2(dys[i], dxs[i]));
  }

  double distance = 0.0;
  s_vec->push_back(distance);
  double fx = xy_points[0].x();
  double fy = xy_points[0].y();
  double nx = 0.0;
  double ny = 0.0;
  for (std::size_t i = 1; i < points_size; ++i) {
    nx = xy_points[i].x();
    ny = xy_points[i].y();
    double end_segment_s =
        std::sqrt((fx - nx) * (fx - nx) + (fy - ny) * (fy - ny));
    s_vec->push_back(end_segment_s + distance);
    distance += end_segment_s;
    fx = nx;
    fy = ny;
  }

  for (std::size_t i = 0; i < points_size; ++i) {
    double xds = 0.0;
    double yds = 0.0;
    if (i == 0) {
      xds = (xy_points[i + 1].x() - xy_points[i].x()) /
            (s_vec->at(i + 1) - s_vec->at(i));
      yds = (xy_points[i + 1].y() - xy_points[i].y()) /
            (s_vec->at(i + 1) - s_vec->at(i));
    } else if (i == points_size - 1) {
      xds = (xy_points[i].x() - xy_points[i - 1].x()) /
            (s_vec->at(i) - s_vec->at(i - 1));
      yds = (xy_points[i].y() - xy_points[i - 1].y()) /
            (s_vec->at(i) - s_vec->at(i - 1));
    } else {
      xds = (xy_points[i + 1].x() - xy_points[i - 1].x()) /
            (s_vec->at(i + 1) - s_vec->at(i - 1));
      yds = (xy_points[i + 1].y() - xy_points[i - 1].y()) /
            (s_vec->at(i + 1) - s_vec->at(i - 1));
    }
    x_over_s_first_derivatives.push_back(xds);
    y_over_s_first_derivatives.push_back(yds);
  }

  for (std::size_t i = 0; i < points_size; ++i) {
    double xdds = 0.0;
    double ydds = 0.0;
    if (i == 0) {
      xdds =
          (x_over_s_first_derivatives[i + 1] - x_over_s_first_derivatives[i]) /
          (s_vec->at(i + 1) - s_vec->at(i));
      ydds =
          (y_over_s_first_derivatives[i + 1] - y_over_s_first_derivatives[i]) /
          (s_vec->at(i + 1) - s_vec->at(i));
    } else if (i == points_size - 1) {
      xdds =
          (x_over_s_first_derivatives[i] - x_over_s_first_derivatives[i - 1]) /
          (s_vec->at(i) - s_vec->at(i - 1));
      ydds =
          (y_over_s_first_derivatives[i] - y_over_s_first_derivatives[i - 1]) /
          (s_vec->at(i) - s_vec->at(i - 1));
    } else {
      xdds = (x_over_s_first_derivatives[i + 1] -
              x_over_s_first_derivatives[i - 1]) /
             (s_vec->at(i + 1) - s_vec->at(i - 1));
      ydds = (y_over_s_first_derivatives[i + 1] -
              y_over_s_first_derivatives[i - 1]) /
             (s_vec->at(i + 1) - s_vec->at(i - 1));
    }
    x_over_s_second_derivatives.push_back(xdds);
    y_over_s_second_derivatives.push_back(ydds);
  }

  for (std::size_t i = 0; i < points_size; ++i) {
    double xds = x_over_s_first_derivatives[i];
    double yds = y_over_s_first_derivatives[i];
    double xdds = x_over_s_second_derivatives[i];
    double ydds = y_over_s_second_derivatives[i];
    double kappa =
        (xds * ydds - yds * xdds) /
        (std::sqrt(xds * xds + yds * yds) * (xds * xds + yds * yds) + 1e-6);
    kappas->push_back(kappa);
  }

  for (std::size_t i = 0; i < points_size; ++i) {
    double dkappa = 0.0;
    if (i == 0) {
      dkappa = (kappas->at(i + 1) - kappas->at(i)) /
               (s_vec->at(i + 1) - s_vec->at(i));
    } else if (i == points_size - 1) {
      dkappa = (kappas->at(i) - kappas->at(i - 1)) /
               (s_vec->at(i) - s_vec->at(i - 1));
    } else {
      dkappa = (kappas->at(i + 1) - kappas->at(i - 1)) /
               (s_vec->at(i + 1) - s_vec->at(i - 1));
    }
    dkappas->push_back(dkappa);
  }
  return true;
}

} // namespace planning
