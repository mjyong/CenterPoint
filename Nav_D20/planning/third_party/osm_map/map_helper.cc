#include "map_helper.h"

namespace lanelet 
{
namespace osm 
{
    
// 计算二维向量叉积
  double crossProduct(const Point2D& a, const Point2D& b) 
  {
    return a.x * b.y - a.y * b.x;
  }
    
// 计算二维向量叉积 (OA × OB)
  double cross(const Point2D& O, const Point2D& A, const Point2D& B) 
  {
    return (A.x - O.x) * (B.y - O.y) - (A.y - O.y) * (B.x - O.x);
  }
    
// 计算两点距离的平方（避免开方运算）
  double distanceSquared(const Point2D& a, const Point2D& b) 
  {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return dx * dx + dy * dy;
  }
    
// 计算向量从a到b
  Point2D vectorFromTo(const Point2D& a, const Point2D& b) 
  {
    return {b.x - a.x, b.y - a.y};
  }
    
// 计算凸包
  std::vector<Point2D> convexHullAndrew(std::vector<Point2D> points) 
  {
    size_t n = points.size();
    if (n <= 3) return points;
        
    // 移除重复点
    std::sort(points.begin(), points.end());
    points.erase(std::unique(points.begin(), points.end()), points.end());
    n = points.size();
    if (n <= 3) return points;
        
    std::vector<Point2D> hull;
    hull.reserve(2 * n); // 预分配空间
        
    // 构建下凸包
    for (size_t i = 0; i < n; i++) 
    {
      while (hull.size() >= 2 && 
                   cross(hull[hull.size() - 2], hull.back(), points[i]) <= 0) 
      {
        hull.pop_back();
      }
      hull.push_back(points[i]);
    }
        
    // 构建上凸包
    size_t lower_size = hull.size();
    for (int i = n - 2; i >= 0; i--) 
    {
      while (hull.size() > lower_size && 
                   cross(hull[hull.size() - 2], hull.back(), points[i]) <= 0) 
      {
        hull.pop_back();
      }
      hull.push_back(points[i]);
    }
        
    // 移除最后一个点（与第一个点重复）
    if (hull.size() > 1) 
    {
      hull.pop_back();
    }
    
    return hull;
  }
    
// 计算多边形的质心
  Point2D computeCentroid(const std::vector<Point2D>& polygon) 
  {
    int n = polygon.size();
    if (n == 0) return {0, 0};  // 空多边形返回(0,0)
    if (n == 1) return polygon[0];  // 单点多边形返回该点
        
    double signedArea = 0.0;
    double centroidX = 0.0;
    double centroidY = 0.0;
        
    for (int i = 0; i < n; ++i) 
    {
      const Point2D& p1 = polygon[i];
            const Point2D& p2 = polygon[(i + 1) % n];  // 下一个顶点（循环）
            
      double cross = crossProduct(p1, p2);
      signedArea += cross;
      centroidX += (p1.x + p2.x) * cross;
      centroidY += (p1.y + p2.y) * cross;
    }
        
    signedArea *= 0.5;  // 实际带符号面积
        
    // 处理退化多边形（面积接近0）
    if (std::fabs(signedArea) < 1e-10) 
    {
      // 计算所有顶点的平均值
      double sumX = 0.0, sumY = 0.0;
      for (const auto& p : polygon) 
      {
        sumX += p.x;
        sumY += p.y;
      }
      return {sumX / n, sumY / n};
    }
        
    double areaFactor = 1.0 / (6.0 * signedArea);  // 1/(6A)
    return {
      centroidX * areaFactor,
      centroidY * areaFactor
    };
  }
    
// 判断点p是否在线段ab上（包括端点）
  bool isPointOnSegment(const Point2D& p, const Point2D& a, const Point2D& b) 
  {
    // 快速排除：检查是否在包围盒外
    if (p.x < std::min(a.x, b.x) || p.x > std::max(a.x, b.x) || 
            p.y < std::min(a.y, b.y) || p.y > std::max(a.y, b.y)) 
    {
      return false;
    }
    
    // 叉积精确判断：检查三点共线
    double cross = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    if (std::fabs(cross) > 1e-10) 
    {
      return false;
    }
    
    // 点积检查：确保点在线段端点之间
    double dot = (p.x - a.x) * (b.x - a.x) + (p.y - a.y) * (b.y - a.y);
    double len2 = (b.x - a.x) * (b.x - a.x) + (b.y - a.y) * (b.y - a.y);
    
    return dot >= 0 && dot <= len2;
  }
    
// 快速判断两个线段是否相交（包括端点相交）
  bool doSegmentsIntersect(const Point2D& p1, const Point2D& p2, const Point2D& q1, const Point2D& q2) 
  {
    // 快速包围盒检查：如果两个线段的包围盒不相交，则线段肯定不相交
    if (std::max(p1.x, p2.x) < std::min(q1.x, q2.x) ||
      std::min(p1.x, p2.x) > std::max(q1.x, q2.x) ||
      std::max(p1.y, p2.y) < std::min(q1.y, q2.y) ||
      std::min(p1.y, p2.y) > std::max(q1.y, q2.y)) 
    {
      return false;
    }
    
    // 计算四个叉积
    Point2D v1 = vectorFromTo(p1, q1);
    Point2D v2 = vectorFromTo(p1, q2);
    Point2D v3 = vectorFromTo(p1, p2);
    
    double cross1 = crossProduct(v3, v1); // (p2-p1) × (q1-p1)
    double cross2 = crossProduct(v3, v2); // (p2-p1) × (q2-p1)
    
    // 如果两个叉积同号，说明q1和q2在p1p2的同一侧，不相交
    if (cross1 * cross2 > 1e-10) 
    {
      return false;
    }
      
    // 同样检查q1q2相对于p1p2的位置
    Point2D w1 = vectorFromTo(q1, p1);
    Point2D w2 = vectorFromTo(q1, p2);
    Point2D w3 = vectorFromTo(q1, q2);
      
    double cross3 = crossProduct(w3, w1); // (q2-q1) × (p1-q1)
    double cross4 = crossProduct(w3, w2); // (q2-q1) × (p2-q1)
      
    // 如果两个叉积同号，说明p1和p2在q1q2的同一侧，不相交
    if (cross3 * cross4 > 1e-10) 
    {
      return false;
    }
      
    // 特殊情况处理：共线情况
    if (std::abs(cross1) < 1e-10 && std::abs(cross2) < 1e-10 &&
      std::abs(cross3) < 1e-10 && std::abs(cross4) < 1e-10) 
    {
      // 线段共线，检查是否有重叠
      return !(std::max(p1.x, p2.x) < std::min(q1.x, q2.x) ||
      std::min(p1.x, p2.x) > std::max(q1.x, q2.x) ||
      std::max(p1.y, p2.y) < std::min(q1.y, q2.y) ||
      std::min(p1.y, p2.y) > std::max(q1.y, q2.y));
    }
      
    return true;
  }
    
// 判断点是否在多边形内部（包括边界）
  bool isPointInPolygon(const Point2D& point, const std::vector<Point2D>& polygon) 
  {
    int n = polygon.size();
    if (n < 3) return false; // 非多边形
        
    bool inside = false;
    Point2D p1 = polygon[0];
        
    for (int i = 1; i <= n; ++i) 
    {
      Point2D p2 = polygon[i % n];
            
      // 快速检查：点是否在当前线段上
      if (isPointOnSegment(point, p1, p2)) 
      {
        return true;
      }
            
      // 射线法核心逻辑
      if ((p1.y > point.y) != (p2.y > point.y)) // 跨越水平线
      {
        // 计算交点横坐标（避免除零）
        if (p2.y != p1.y) 
        {
          double xIntersect = (p2.x - p1.x) * (point.y - p1.y) / (p2.y - p1.y) + p1.x;
                    
          // 交点在点右侧且非顶点（顶点已处理）
          if (point.x < xIntersect) 
          {
            inside = !inside;
          }
        }
      }
      p1 = p2; // 移动到下一条边
    }
        
    return inside;
  }
    
// 判断多边形是否为凸包
  bool isConvexPolygon(const std::vector<Point2D>& polygon) 
  {
    int n = polygon.size();
    if (n < 3) return false; // 至少需要3个点才能构成多边形
        
    // 初始化标志
    bool has_positive = false;
    bool has_negative = false;
        
    for (int i = 0; i < n; i++) 
    {
      // 获取三个连续点
      const Point2D& a = polygon[i];
      const Point2D& b = polygon[(i + 1) % n];
      const Point2D& c = polygon[(i + 2) % n];
            
      // 计算向量AB和BC
      Point2D ab = b - a;
      Point2D bc = c - b;
            
      // 计算叉积
      double cross = crossProduct(ab, bc);
            
      // 检查叉积符号
      if (cross > 1e-10) // 正数，考虑浮点误差
      {
        has_positive = true;
      } else if (cross < -1e-10) // 负数，考虑浮点误差
      {
        has_negative = true;
      }
            
      // 如果同时存在正负叉积，则不是凸包
      if (has_positive && has_negative) 
      {
        return false;
      }
    }
        
    // 所有叉积同号（或部分为零），是凸包
    return true;
  }
    
// 优化版本：判断多边形是否为凸包
  bool isConvexPolygonFast(const std::vector<Point2D>& polygon) 
  {
    int n = polygon.size();
    if (n < 3) return false;
        
    int sign = 0; // 0-未确定，1-正，-1-负
        
    for (int i = 0; i < n; i++) 
    {
      const Point2D& a = polygon[i];
      const Point2D& b = polygon[(i + 1) % n];
      const Point2D& c = polygon[(i + 2) % n];
            
      Point2D ab = b - a;
      Point2D bc = c - b;
            
      double cross = crossProduct(ab, bc);
            
      // 忽略接近零的叉积（共线情况）
      if (std::fabs(cross) < 1e-10) continue;
            
      int current_sign = (cross > 0) ? 1 : -1;
            
      if (sign == 0) 
      {
        // 第一次遇到非零叉积，记录符号
        sign = current_sign;
      } else if (sign != current_sign) 
      {
        // 发现不一致的符号，不是凸包
        return false;
      }
    }
        
    // 所有非零叉积符号一致（或全部接近零），是凸包
    return true;
  }
    
// 计算多边形的轴向包围盒（AABB）
  void computeAABB(const std::vector<Point2D>& polygon, Point2D& minCorner, Point2D& maxCorner) 
  {
    if (polygon.empty()) 
    {
      minCorner = Point2D(0, 0);
      maxCorner = Point2D(0, 0);
      return;
    }
    
    // 初始化最小和最大值
    double minX = polygon[0].x;
    double maxX = polygon[0].x;
    double minY = polygon[0].y;
    double maxY = polygon[0].y;
    
    // 遍历所有顶点，找到边界
    for (const auto& p : polygon) 
    {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    
    minCorner = Point2D(minX, minY);
    maxCorner = Point2D(maxX, maxY);
  }
    
// 计算点到直线的距离
  double pointToLineDistance(const Point2D& p, const Point2D& lineStart, const Point2D& lineEnd) 
  {
    double numerator = std::abs((lineEnd.y - lineStart.y) * p.x - 
                      (lineEnd.x - lineStart.x) * p.y + 
                      lineEnd.x * lineStart.y - 
                      lineEnd.y * lineStart.x);
    double denominator = std::sqrt(std::pow(lineEnd.y - lineStart.y, 2) + 
                         std::pow(lineEnd.x - lineStart.x, 2));
    return numerator / denominator;
  }
    
// 计算凸多边形的最小面积包围矩形（旋转矩形）
  RotatedRect computeMinAreaRect(const std::vector<Point2D>& convexHull) 
  {
    int n = convexHull.size();
    if (n < 3) 
    {
      return {Point2D(0, 0), Point2D(0, 0), 0};
    }
    
    // 初始化变量
    double minArea = std::numeric_limits<double>::max();
    RotatedRect result;
    
    // 旋转卡壳法：对每条边作为矩形的一边进行尝试
    int j = 1, k = 1, l = 1;
    for (int i = 0; i < n; i++) 
    {
      Point2D p1 = convexHull[i];
      Point2D p2 = convexHull[(i + 1) % n];
    
      // 计算边的方向向量
      Point2D edgeDir(p2.x - p1.x, p2.y - p1.y);
      double edgeLength = std::sqrt(edgeDir.x * edgeDir.x + edgeDir.y * edgeDir.y);
      Point2D edgeUnit(edgeDir.x / edgeLength, edgeDir.y / edgeLength);
    
      // 计算垂直方向
      Point2D perpendicular(-edgeUnit.y, edgeUnit.x);
    
      // 找到最远的点（在垂直方向上）
      while (true) 
      {
        Point2D next = convexHull[(j + 1) % n];
        double proj = (next.x - p1.x) * perpendicular.x + (next.y - p1.y) * perpendicular.y;
        double current = (convexHull[j].x - p1.x) * perpendicular.x + (convexHull[j].y - p1.y) * perpendicular.y;
        if (proj > current) 
        {
          j = (j + 1) % n;
        } 
        else 
        {
          break;
        }
      }
    
      // 找到最远的点（在边缘方向上）
      while (true) 
      {
        Point2D next = convexHull[(k + 1) % n];
        double proj = (next.x - p1.x) * edgeUnit.x + (next.y - p1.y) * edgeUnit.y;
        double current = (convexHull[k].x - p1.x) * edgeUnit.x + (convexHull[k].y - p1.y) * edgeUnit.y;
        if (proj > current) 
        {
          k = (k + 1) % n;
        } 
        else 
        {
          break;
        }
      }
    
      // 找到最远的点（在负垂直方向上）
      while (true) 
      {
        Point2D next = convexHull[(l + 1) % n];
        double proj = (next.x - p1.x) * (-perpendicular.x) + (next.y - p1.y) * (-perpendicular.y);
        double current = (convexHull[l].x - p1.x) * (-perpendicular.x) + (convexHull[l].y - p1.y) * (-perpendicular.y);
        if (proj > current) 
        {
          l = (l + 1) % n;
        } 
        else 
        {
          break;
        }
      }
    
      // 计算矩形的宽度和高度
      double width = pointToLineDistance(convexHull[j], p1, p2);
      double height = pointToLineDistance(convexHull[k], p1, p2) + pointToLineDistance(convexHull[l], p1, p2);
    
      // 计算面积
      double area = width * height;
    
      // 更新最小面积矩形
      if (area < minArea) 
      {
        minArea = area;
        result.center.x = (p1.x + p2.x) / 2.0;
        result.center.y = (p1.y + p2.y) / 2.0;
        result.size.x = width;
        result.size.y = height;
        result.angle = std::atan2(edgeDir.y, edgeDir.x);
      }
    }
    
    return result;
  }
    
// 快速凸包算法（Andrew's monotone chain algorithm）
  std::vector<Point2D> computeConvexHull(std::vector<Point2D> points) 
  {
    if (points.size() <= 3) return points;
    
    // 按x坐标排序，如果x相同则按y排序
    std::sort(points.begin(), points.end(), [](const Point2D& a, const Point2D& b) 
    {
      return a.x < b.x || (a.x == b.x && a.y < b.y);
    });
    
    std::vector<Point2D> hull;
    
    // 构建下凸包
    for (const auto& p : points) 
    {
      while (hull.size() >= 2) 
      {
        const Point2D& a = hull[hull.size() - 2];
        const Point2D& b = hull[hull.size() - 1];
        double cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if (cross <= 0) break; // 如果是顺时针或共线，则移除
        hull.pop_back();
      }
      hull.push_back(p);
    }
    
    // 构建上凸包
    size_t lower_size = hull.size();
    for (auto it = points.rbegin(); it != points.rend(); ++it) 
    {
      const Point2D& p = *it;
      while (hull.size() > lower_size) 
      {
        const Point2D& a = hull[hull.size() - 2];
        const Point2D& b = hull[hull.size() - 1];
        double cross = (b.x - a.x) * (p.y - a.y) - (b.y - a.y) * (p.x - a.x);
        if (cross <= 0) break; // 如果是顺时针或共线，则移除
        hull.pop_back();
      }
      hull.push_back(p);
    }
    
    // 移除最后一个点（与第一个点重复）
    if (hull.size() > 1) hull.pop_back();
    
    return hull;
  }
    
// 计算任意多边形的最小包围矩形
// 如果是凹多边形，先计算凸包，再计算最小面积矩形
  RotatedRect computeBoundingRect(const std::vector<Point2D>& polygon, bool useAABB) 
  {
    if (useAABB) 
    {
      // 使用AABB
      Point2D minCorner, maxCorner;
      computeAABB(polygon, minCorner, maxCorner);
    
      RotatedRect result;
      result.center.x = (minCorner.x + maxCorner.x) / 2.0;
      result.center.y = (minCorner.y + maxCorner.y) / 2.0;
      result.size.x = maxCorner.x - minCorner.x;
      result.size.y = maxCorner.y - minCorner.y;
      result.angle = 0;
      return result;
    } else 
    {
      // 使用最小面积矩形（需要凸包）
      std::vector<Point2D> convexHull = computeConvexHull(polygon);
      return computeMinAreaRect(convexHull);
    }
  }
    
    
    
    
    
    
    
    
    
    
    
    
    }
    }