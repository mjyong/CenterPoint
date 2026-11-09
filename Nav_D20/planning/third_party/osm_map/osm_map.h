#ifndef OSM_MAP_OSMMAP_H
#define OSM_MAP_OSMMAP_H

#include <vector>
#include <string>

namespace lanelet 
{

enum class SemanticType
{
    Unknown = 0,
    Footway = 1, //人行道
    Steps = 2,  //台阶
    Highway = 3, //车道区域
    Crossing = 4, //路口
    Crosswalk = 5, //人行横道
    Centerline = 6, //车道中心线
    VirtualLine = 7, //虚拟线
    Fence = 8, //电子围栏
    ProhibitedArea = 9, //禁行区域
    ReadyPoint = 11, //待命点
    AccessPoint = 12, //门禁点
    WallPoint = 13, //墙面上沿
    LampPoint = 14, //路灯
    TrafficLightPoint = 15 //交通灯
};

enum class GeometryType 
{
    Unknown = 0,
    Polygon = 1, //多边形
    Curve = 2,   //曲线
    Line = 3,    //线段
    KPoint = 4   //点
};

struct Point2D 
{
    double x{0.};
    double y{0.};

    Point2D(double x = 0, double y = 0) : x(x), y(y) {}
    Point2D operator-(const Point2D& other) const 
    {
        return Point2D(x - other.x, y - other.y);
    }
    bool operator<(const Point2D& other) const 
    {
        return x < other.x || (x == other.x && y < other.y);
    }
    bool operator==(const Point2D& other) const 
    {
        return std::abs(x - other.x) < 1e-10 && std::abs(y - other.y) < 1e-10;
    }
};

struct Point3D 
{
    double x{0.};
    double y{0.};
    double z{0.};
};

using LineString3D = std::vector<Point3D>;

struct KeyPoint 
{
    SemanticType type;
    GeometryType geometry;
    LineString3D points;
    void reset()
    {
        type = SemanticType::Unknown;
        geometry = GeometryType::Unknown;
        points.clear();
    }
};

using KeyPoints = std::vector<KeyPoint>;

struct Map 
{
    KeyPoints keyPoints;
    void reset()
    {
        keyPoints.clear();
    }
};

}
#endif
