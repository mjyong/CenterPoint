#ifndef OSM_MAP_MAPHELPER_H
#define OSM_MAP_MAPHELPER_H

#include "osm_map.h"
#include <vector>
#include <cmath>
#include <algorithm>

namespace lanelet 
{
namespace osm 
{

  // 计算二维向量叉积
  double crossProduct(const Point2D& a, const Point2D& b);

  // 计算二维向量叉积 (OA × OB)
  double cross(const Point2D& O, const Point2D& A, const Point2D& B);

  // 计算两点距离的平方（避免开方运算）
  double distanceSquared(const Point2D& a, const Point2D& b);

  // 计算向量从a到b
  Point2D vectorFromTo(const Point2D& a, const Point2D& b);

  std::vector<Point2D> convexHullAndrew(std::vector<Point2D> points);

  // 计算多边形的质心
  Point2D computeCentroid(const std::vector<Point2D>& polygon);

  // 计算多边形的质心
  Point2D computeCentroid(const std::vector<Point2D>& polygon);

  // 判断点p是否在线段ab上（包括端点）
  bool isPointOnSegment(const Point2D& p, const Point2D& a, const Point2D& b);

  // 快速判断两个线段是否相交（包括端点相交）
  bool doSegmentsIntersect(const Point2D& p1, const Point2D& p2, const Point2D& q1, const Point2D& q2); 

  // 判断点是否在多边形内部（包括边界）
  bool isPointInPolygon(const Point2D& point, const std::vector<Point2D>& polygon);

  // 判断多边形是否为凸包
  bool isConvexPolygon(const std::vector<Point2D>& polygon);

  // 优化版本：判断多边形是否为凸包
  bool isConvexPolygonFast(const std::vector<Point2D>& polygon);
  // 计算多边形的轴向包围盒（AABB）
  void computeAABB(const std::vector<Point2D>& polygon, Point2D& minCorner, Point2D& maxCorner);

  // 计算多边形的最小包围矩形（包括旋转的情况）
  // 使用旋转卡壳法，适用于凸多边形
  struct RotatedRect {
    Point2D center;
    Point2D size;
    double angle; // 旋转角度（弧度）
  };

  // 计算点到直线的距离
  double pointToLineDistance(const Point2D& p, const Point2D& lineStart, const Point2D& lineEnd);

  // 计算凸多边形的最小面积包围矩形（旋转矩形）
  RotatedRect computeMinAreaRect(const std::vector<Point2D>& convexHull);

  // 快速凸包算法（Andrew's monotone chain algorithm）
  std::vector<Point2D> computeConvexHull(std::vector<Point2D> points);

  // 计算任意多边形的最小包围矩形
  // 如果是凹多边形，先计算凸包，再计算最小面积矩形
  RotatedRect computeBoundingRect(const std::vector<Point2D>& polygon, bool useAABB = false);


}
}
#endif
