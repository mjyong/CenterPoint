#include "osm_parser.h"

namespace lanelet {
namespace osm {

std::string buildErrorMessage(const std::string& errorIntro, const Errors& errors) {
  if (errors.empty()) {
    return std::string{};
  }
  std::string message{errorIntro};
  for (const auto& error : errors) {
    message.append(" - " + error);
  }
  return message;
}

void OsmParser::reset()
{
    map_.reset();
    semanticDict_.clear();
    semanticIdx_.clear();
    typeRegister_.clear();
    semanticRegister_.clear();
}

Map& OsmParser::getMap()
{
  return map_;
}

Map OsmParser::getMap(std::vector<int>& types)
{
  if(types.empty())
  {
    return map_;
  }

  Map result;
  for(auto& type : types)
  {
    auto raw_areas = selectSemantic(static_cast<SemanticType>(type));
    if(raw_areas.empty())
    {
      continue;
    }
    for(auto& area_3d : raw_areas)
    {
      KeyPoint kp;
      kp.type = area_3d.type;
      kp.geometry = area_3d.geometry;
      kp.points = area_3d.points;
      result.keyPoints.push_back(kp);
    }
  }
  return result;
}

void OsmParser::insertSemanticDict(const Id& id, const KeyPoint& area)
{
  semanticDict_.insert({id, area});
}

void OsmParser::insertSemanticIdx(const SemanticType& type, const Id& id)
{
  semanticIdx_[type].push_back(id);
}

void OsmParser::printOsmFile(const File& file) {
  std::cout << "Nodes: " << file.nodes.size() << std::endl;
  std::cout << "Ways: " << file.ways.size() << std::endl;
  std::cout << "Relations: " << file.relations.size() << std::endl;
}

std::pair<SemanticType, GeometryType> OsmParser::fromSubtype(const std::string& subtype)
{
  std::pair<SemanticType, GeometryType> result{SemanticType::Unknown, GeometryType::Unknown};
  auto it = semanticRegister_.find(subtype);
  if(it != semanticRegister_.end())
  {
    result = it->second;
  }
  return result;
}

std::pair<Id, KeyPoint> OsmParser::fromPolygon(const Relation& relation)
{
  std::pair<Id, KeyPoint> result;
  result.first = relation.id;
  std::cout << relation.attributes.at(keyword::Subtype) << " : ";
  if(relation.members.size() == 1)
  {
    auto& member = relation.members.front();
    if(member.first == keyword::Outer && member.second->type() == keyword::Way)
    {
      Way* way = dynamic_cast<Way*>(member.second);
      auto& nodes = way->nodes;
      for(size_t i = 0; i < nodes.size()-1; i++)
      {
        result.second.points.push_back(Point3D{std::stod(nodes[i]->attributes.at(keyword::LocalX)), std::stod(nodes[i]->attributes.at(keyword::LocalY)), 0});
        std::cout << nodes[i]->id << " " << std::stod(nodes[i]->attributes.at(keyword::LocalX)) << " " << std::stod(nodes[i]->attributes.at(keyword::LocalY)) << std::endl;
      }
      independentWaySet_.erase(way->id);
    }
  }
  else
  {
    std::cerr << "relation " << relation.id << " has more than one member" << std::endl;
    result.second.reset();
    return result;
  }
  auto info = fromSubtype(relation.attributes.at(keyword::Subtype));
  result.second.type = info.first;
  result.second.geometry = info.second;
  return result;
}

std::pair<Ids, KeyPoints> OsmParser::fromLanelet(const Relation& relation)
{
  std::pair<Ids, KeyPoints> result;
  auto& members = relation.members;
  if(members.size() >= 3)
  {
    Way* leftWay = nullptr;
    Way* rightWay = nullptr;
    Way* centerlineWay = nullptr;
    for(auto& member : members)
    {
      if(member.second->type() == keyword::Way)
      {
        Way* way = dynamic_cast<Way*>(member.second);
        if(member.first == keyword::Left)
        {
          leftWay = way;
          independentWaySet_.erase(way->id);
        }
        else if(member.first == keyword::Right)
        {
          rightWay = way;
          independentWaySet_.erase(way->id);
        }
        else if(member.first == keyword::Centerline)
        {
          centerlineWay = way;
          independentWaySet_.erase(way->id);
        }
        else
        {
          std::cerr << "relation " << relation.id << " has unknown member " << member.first << std::endl;
        }
      }
    }
    if(centerlineWay != nullptr)
    {
      KeyPoint area;
      auto& nodes = centerlineWay->nodes;
      for(size_t i = 0; i < nodes.size(); i++)
      {
        if(nodes[i]->attributes.find(keyword::LocalX) == nodes[i]->attributes.end() || nodes[i]->attributes.find(keyword::LocalY) == nodes[i]->attributes.end()){
          std::cerr << "way " << centerlineWay->id << " has unknown local x or y" << std::endl;
          break;
        }
        area.points.push_back(Point3D{std::stod(nodes[i]->attributes.at(keyword::LocalX)), std::stod(nodes[i]->attributes.at(keyword::LocalY)), 0});
      }
      area.type = SemanticType::Centerline;
      area.geometry = GeometryType::Curve;
      result.second.push_back(area);
      result.first.push_back(centerlineWay->id);
    }
    if(leftWay != nullptr && rightWay != nullptr)
    {
      KeyPoint area;
      auto& leftNodes = leftWay->nodes;
      auto& rightNodes = rightWay->nodes;
      if(leftNodes.size() < 2 || rightNodes.size() < 2)
      {
        return result;
      }
      if(leftNodes.front()->attributes.find(keyword::LocalX) == leftNodes.front()->attributes.end()
        || leftNodes.front()->attributes.find(keyword::LocalY) == leftNodes.front()->attributes.end()
        || rightNodes.front()->attributes.find(keyword::LocalX) == rightNodes.front()->attributes.end()
        || rightNodes.front()->attributes.find(keyword::LocalY) ==  rightNodes.front()->attributes.end())
      {
        return result;
      }
      for(auto& nd : leftNodes)
      {
        area.points.push_back(Point3D{std::stod(nd->attributes.at(keyword::LocalX)), std::stod(nd->attributes.at(keyword::LocalY)), 0});
      }
      Point2D leftFront = {std::stod(leftNodes.front()->attributes.at(keyword::LocalX)), std::stod(leftNodes.front()->attributes.at(keyword::LocalY))};
      Point2D leftBack = {std::stod(leftNodes.back()->attributes.at(keyword::LocalX)), std::stod(leftNodes.back()->attributes.at(keyword::LocalY))};
      Point2D rightFront = {std::stod(rightNodes.front()->attributes.at(keyword::LocalX)), std::stod(rightNodes.front()->attributes.at(keyword::LocalY))};
      Point2D rightBack = {std::stod(rightNodes.back()->attributes.at(keyword::LocalX)), std::stod(rightNodes.back()->attributes.at(keyword::LocalY))};
      if(doSegmentsIntersect(leftFront, rightBack, rightFront, leftBack))
      {
        for(int i = rightNodes.size()-1; i >= 0; i--)
        {
          area.points.push_back(Point3D{std::stod(rightNodes[i]->attributes.at(keyword::LocalX)), std::stod(rightNodes[i]->attributes.at(keyword::LocalY)), 0});
        }
      }
      else
      {
        for(auto& nd : rightNodes)
        {
          area.points.push_back(Point3D{std::stod(nd->attributes.at(keyword::LocalX)), std::stod(nd->attributes.at(keyword::LocalY)), 0});
        }
        std::cout << "## way " << leftWay->id << " and " << rightWay->id << " do not intersect" << std::endl;
      }
      area.type = SemanticType::Highway;
      area.geometry = GeometryType::Polygon;
      result.second.push_back(area);
      result.first.push_back(relation.id);
    }
  }
  else
  {
    std::cerr << "relation " << relation.id << " has less than three members" << std::endl;
  }
  return result;
}

std::pair<Id, KeyPoint> OsmParser::fromThreeD(const Way& way)
{
  std::pair<Id, KeyPoint> result;
  result.first = way.id;
  size_t numNodes = way.nodes.size();
  if(numNodes < 2)
  {
    std::cerr << "way " << way.id << " has less than two nodes" << std::endl;
    result.second.reset();
    return result;
  }
  auto kpType = way.attributes.at(keyword::ThreeD);
  auto it = semanticRegister_.find(kpType);
  if(it != semanticRegister_.end())
  {
    result.second.type = it->second.first;
    result.second.geometry = it->second.second;
    switch (it->second.second)
    {
    case GeometryType::Curve:
    {
      for(auto& node : way.nodes)
      {
        if(node->attributes.find(keyword::LocalX) == node->attributes.end() || node->attributes.find(keyword::LocalY) == node->attributes.end())
        {
          std::cerr << "way " << way.id << " has unknown local x or y" << std::endl;
          result.second.reset();
          return result;
        }
        result.second.points.push_back(Point3D{std::stod(node->attributes.at(keyword::LocalX)), std::stod(node->attributes.at(keyword::LocalY)), node->point.ele});
      }
      std::cout << kpType << " : " << way.id << " " << result.second.points.size() << std::endl;
    }
    break;
    case GeometryType::KPoint:
    {
      double sumX = 0;
      double sumY = 0;
      double sumZ = 0;
      for(auto& node : way.nodes)
      {
        if(node->attributes.find(keyword::LocalX) == node->attributes.end() || node->attributes.find(keyword::LocalY) == node->attributes.end())
        {
          std::cerr << "way " << way.id << " has unknown local x or y" << std::endl;
          result.second.reset();
          return result;
        }
        sumX += std::stod(node->attributes.at(keyword::LocalX));
        sumY += std::stod(node->attributes.at(keyword::LocalY));
        sumZ += node->point.ele;
      }
      std::cout << kpType << " : " << way.id << " " << sumX/numNodes << " " << sumY/numNodes << " " << sumZ/numNodes << std::endl;
      result.second.points.push_back(Point3D{sumX/numNodes, sumY/numNodes, sumZ/numNodes});
    }
    break;
    default:
      break;
    }
  }
  else
  {
    std::cerr << "way " << way.id << " has unknown 3d type " << kpType << std::endl;
  }
  return result;
}

std::pair<Id, KeyPoint> OsmParser::fromR11(const Way& way)
{
  std::pair<Id, KeyPoint> result;
  result.first = way.id;
  size_t numNodes = way.nodes.size();
  if(numNodes < 1)
  {
    std::cerr << "way " << way.id << " has less than one nodes" << std::endl;
    result.second.reset();
    return result;
  }
  auto kpType = way.attributes.at(keyword::Type);
  auto it = semanticRegister_.find(kpType);
  if(it != semanticRegister_.end())
  {
    result.second.type = it->second.first;
    result.second.geometry = it->second.second;
    auto subtype = way.attributes.find(keyword::SubType);
    if(subtype == way.attributes.end())
    {
      subtype = way.attributes.find(keyword::Subtype);
    }
    if(subtype != way.attributes.end())
    {
      if(subtype->second == "polygon")
      {
        result.second.geometry = GeometryType::Polygon;
      }
      else if(subtype->second == "line")
      {
        result.second.geometry = GeometryType::Line;
      }
      else if(subtype->second == "curve")
      {
        result.second.geometry = GeometryType::Curve;
      }
    }
    switch (result.second.geometry)
    {
    case GeometryType::Polygon:
    case GeometryType::Curve:
    {
      for(auto& node : way.nodes)
      {
        if(node->attributes.find(keyword::LocalX) == node->attributes.end() || node->attributes.find(keyword::LocalY) == node->attributes.end())
        {
          std::cerr << "way " << way.id << " has unknown local x or y" << std::endl;
          result.second.reset();
          return result;
        }
        result.second.points.push_back(Point3D{std::stod(node->attributes.at(keyword::LocalX)), std::stod(node->attributes.at(keyword::LocalY)), node->point.ele});
      }
      std::cout << kpType << " : " << way.id << " " << result.second.points.size() << std::endl;
    }
    break;
    case GeometryType::KPoint:
    {
      double sumX = 0;
      double sumY = 0;
      double sumZ = 0;
      for(auto& node : way.nodes)
      {
        if(node->attributes.find(keyword::LocalX) == node->attributes.end() || node->attributes.find(keyword::LocalY) == node->attributes.end())
        {
          std::cerr << "way " << way.id << " has unknown local x or y" << std::endl;
          result.second.reset();
          return result;
        }
        sumX += std::stod(node->attributes.at(keyword::LocalX));
        sumY += std::stod(node->attributes.at(keyword::LocalY));
        sumZ += node->point.ele;
      }
      std::cout << kpType << " : " << way.id << " " << sumX/numNodes << " " << sumY/numNodes << " " << sumZ/numNodes << std::endl;
      result.second.points.push_back(Point3D{sumX/numNodes, sumY/numNodes, sumZ/numNodes});
    }
    break;
    default:
    {
      for(auto& node : way.nodes)
      {
        if(node->attributes.find(keyword::LocalX) == node->attributes.end() || node->attributes.find(keyword::LocalY) == node->attributes.end())
        {
          std::cerr << "way " << way.id << " has unknown local x or y" << std::endl;
          result.second.reset();
          return result;
        }
        result.second.points.push_back(Point3D{std::stod(node->attributes.at(keyword::LocalX)), std::stod(node->attributes.at(keyword::LocalY)), node->point.ele});
      }
      std::cout << kpType << " : " << way.id << " " << result.second.points.size() << std::endl;
    }
    break;
    }
  }
  else
  {
    std::cerr << "way " << way.id << " has unknown 3d type " << kpType << std::endl;
  }
  return result;
}

void OsmParser::fromOsmFile(const File& file) 
{
  independentWaySet_.clear();
  std::copy(file.waysIds.begin(), file.waysIds.end(), std::inserter(independentWaySet_, independentWaySet_.end()));
  for(auto& relation : file.relations)
  {
    if(relation.second.attributes.find(keyword::Type) != relation.second.attributes.end())
    {
      auto& relationType = relation.second.attributes.at(keyword::Type);
      if(relationType == keyword::Polygon)
      {
        auto area = fromPolygon(relation.second);
        map_.keyPoints.push_back(area.second);
        insertSemanticDict(area.first, area.second);
        insertSemanticIdx(area.second.type, area.first);
      }
      else if(relationType == keyword::Lanelet)
      {
        auto lanes = fromLanelet(relation.second);
        map_.keyPoints.insert(map_.keyPoints.end(), lanes.second.begin(), lanes.second.end());
        for(size_t i = 0; i < lanes.first.size(); i++)
        {
          insertSemanticDict(lanes.first[i], lanes.second[i]);
          insertSemanticIdx(lanes.second[i].type, lanes.first[i]);
        }
      }
      else
      {
        std::cerr << "relation " << relation.second.id << " has unknown type " << relationType << std::endl;
      }
    }
    else
    {
      std::cerr << "relation " << relation.second.id << " has no type" << std::endl;
    }
  }
  std::cout << "## independent ways: " << independentWaySet_.size() << std::endl;
  for(auto& wayId : independentWaySet_)
  {
    if(file.ways.at(wayId).attributes.find(keyword::ThreeD) != file.ways.at(wayId).attributes.end())
    {
      auto kp = fromThreeD(file.ways.at(wayId));
      map_.keyPoints.push_back(kp.second);
      insertSemanticDict(kp.first, kp.second);
      insertSemanticIdx(kp.second.type, kp.first);
    }
    else if(file.ways.at(wayId).attributes.find(keyword::Type) != file.ways.at(wayId).attributes.end())
    {
      auto kp = fromR11(file.ways.at(wayId));
      map_.keyPoints.push_back(kp.second);
      insertSemanticDict(kp.first, kp.second);
      insertSemanticIdx(kp.second.type, kp.first);
    }
    else
    {
      std::cerr << "way " << wayId << " has no type" << std::endl;
    }
  }
}

bool OsmParser::loadRegister(const std::string& register_file)
{
  tinyxml2::XMLDocument doc;
  const auto result = doc.LoadFile(register_file.c_str());
  if (result != tinyxml2::XML_SUCCESS)
  {
    std::cerr << "Errors occured while parsing register file: "
              << doc.ErrorStr() << std::endl;
    return false;
  }
  else
  {
    std::cout << "register file load success" << std::endl;
  }

  for (auto* item = doc.FirstChildElement(keyword::Semantic); item != nullptr;
       item = item->NextSiblingElement(keyword::Semantic))
  {
      const std::string semaName = item->Attribute(keyword::Name) == nullptr
                                       ? ""
                                       : item->Attribute(keyword::Name);
      const auto semaType = item->IntAttribute(keyword::Type);
      const auto semaGeom = item->IntAttribute(keyword::Geometry);
      const std::string semaDesc = item->Attribute(keyword::Description) == nullptr
                                       ? ""
                                       : item->Attribute(keyword::Description);
      semanticRegister_.emplace(semaName, std::make_pair(static_cast<SemanticType>(semaType), static_cast<GeometryType>(semaGeom)));
      typeRegister_.emplace(static_cast<SemanticType>(semaType), semaDesc);
      std::cout << "register: " << semaName << " " << semaType << " " << semaGeom << " " << semaDesc << std::endl;
  }
  return true;
}

void OsmParser::loadDefaultRegister()
{
  semanticRegister_.emplace(
      "prohibited_area",
      std::make_pair(SemanticType::ProhibitedArea, GeometryType::Polygon));
  semanticRegister_.emplace(
      "fence", std::make_pair(SemanticType::Fence, GeometryType::Polygon));
  semanticRegister_.emplace(
      "virtual_line",
      std::make_pair(SemanticType::VirtualLine, GeometryType::Line));
}

bool OsmParser::parseOsmFile(const std::string& osm_file)
{
  // read osm file
  tinyxml2::XMLDocument doc;
  const auto osmFile = doc.LoadFile(osm_file.c_str());
  if (osmFile != tinyxml2::XML_SUCCESS)
  {
    std::cerr << "Errors occured while parsing osm file: " << doc.ErrorStr()
              << std::endl;
    return false;
  }
  else
  {
    std::cout << "osm file load success" << std::endl;
  }
  Errors osmReadErrors;
  auto file = read(doc, &osmReadErrors);
  printOsmFile(file);
  fromOsmFile(file);
  std::string errorOutput = buildErrorMessage("Errors ocurred while parsing Lanelet Map:", osmReadErrors);
  if(!errorOutput.empty())
  {
    std::cout << errorOutput << std::endl;
    return false;
  }

  return true;
}

bool OsmParser::parse(const std::string& osm_file)
{
  reset();
  loadDefaultRegister();
  return parseOsmFile(osm_file);
}

bool OsmParser::parse(const std::string& osm_file,
                      const std::string& register_file)
{
  reset();
  if(!loadRegister(register_file))
  {
    return false;
  }
  return parseOsmFile(osm_file);
}

KeyPoints OsmParser::selectSemantic(const SemanticType& type)
{
  KeyPoints result;
  bool isExist = semanticIdx_.find(type) != semanticIdx_.end();
  if(isExist)
  {
    for(auto& id : semanticIdx_.at(type))
    {
      result.push_back(semanticDict_.at(id));
    }
  }
  else
  {
    std::cerr << "Semantic type : " << static_cast<int>(type) << " does not exist" << std::endl;
  }
  return result;
}

}  // namespace osm
}  // namespace lanelet
