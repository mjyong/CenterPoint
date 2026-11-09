#ifndef OSM_MAP_OSMPARSER_H
#define OSM_MAP_OSMPARSER_H

#include "osm_file.h"
#include "osm_map.h"
#include "map_helper.h"
#include <iostream>
#include <string>
#include <set>

namespace lanelet 
{
namespace osm 
{

class OsmParser 
{
public:
    OsmParser() = default;
    OsmParser(OsmParser&& rhs) noexcept = default;
    OsmParser& operator=(OsmParser&& rhs) noexcept = default;
    OsmParser(const OsmParser& rhs) = delete;
    OsmParser& operator=(const OsmParser& rhs) = delete;
    ~OsmParser() noexcept = default;
    bool parse(const std::string& osm_file);
    bool parse(const std::string& osm_file, const std::string& register_file);
    Map& getMap();
    Map getMap(std::vector<int>& types);
    KeyPoints selectSemantic(const SemanticType& type);
    void reset();
    

private:
    void printOsmFile(const File& file);
    void fromOsmFile(const File& file);
    std::pair<Id, KeyPoint> fromPolygon(const Relation& relation);
    std::pair<Ids, KeyPoints> fromLanelet(const Relation& relation);
    std::pair<Id, KeyPoint> fromThreeD(const Way& way);
    std::pair<Id, KeyPoint> fromR11(const Way& way);
    std::pair<SemanticType, GeometryType> fromSubtype(const std::string& subtype);
    void insertSemanticDict(const Id& id, const KeyPoint& area);
    void insertSemanticIdx(const SemanticType& type, const Id& id);
    bool loadRegister(const std::string& register_file);
    void loadDefaultRegister();
    bool parseOsmFile(const std::string& osm_file);

private:
    IdSet independentWaySet_;
    Map map_;
    std::map<Id, KeyPoint> semanticDict_;
    std::map<SemanticType, Ids> semanticIdx_;
    std::map<SemanticType, std::string> typeRegister_;
    std::map<std::string, std::pair<SemanticType, GeometryType>> semanticRegister_;

};
}  // namespace osm
}  // namespace lanelet
#endif
