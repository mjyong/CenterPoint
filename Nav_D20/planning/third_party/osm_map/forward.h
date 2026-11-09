#ifndef OSM_MAP_FORWARD_H
#define OSM_MAP_FORWARD_H

#include <vector>

namespace lanelet 
{

using Id = long long;
constexpr Id InvalId = 0;
using Ids = std::vector<Id>;

//! A raw GPS point
class GPSPoint 
{
public:
  double lat{0.};  //! lat according to WGS84
  double lon{0.};  //! lon according to WGS84
  double ele{0.};  //! elevation according to WGS84 (m)
};

namespace keyword 
{
  constexpr const char* Osm = "osm";
  constexpr const char* Tag = "tag";
  constexpr const char* Key = "k";
  constexpr const char* Value = "v";
  constexpr const char* Node = "node";
  constexpr const char* Way = "way";
  constexpr const char* Relation = "relation";
  constexpr const char* Member = "member";
  constexpr const char* Role = "role";
  constexpr const char* Type = "type";
  constexpr const char* Nd = "nd";
  constexpr const char* Ref = "ref";
  constexpr const char* Id = "id";
  constexpr const char* Lat = "lat";
  constexpr const char* Lon = "lon";
  constexpr const char* Version = "version";
  constexpr const char* Visible = "visible";
  constexpr const char* Elevation = "ele";
  constexpr const char* Action = "action";
  constexpr const char* Delete = "delete";
  
  constexpr const char* Polygon = "multipolygon";
  constexpr const char* Subtype = "subtype";
  constexpr const char* SubType = "subType";
  constexpr const char* Lanelet = "lanelet";
  constexpr const char* Left = "left";
  constexpr const char* Right = "right";
  constexpr const char* Centerline = "centerline";
  constexpr const char* LocalX = "local_x";
  constexpr const char* LocalY = "local_y";
  constexpr const char* Outer = "outer";
  constexpr const char* ThreeD = "3d";
  constexpr const char* Semantic = "semantic";
  constexpr const char* Name = "name";
  constexpr const char* Geometry = "geometry";
  constexpr const char* Description = "description";
  }  // namespace keyword

}
#endif
