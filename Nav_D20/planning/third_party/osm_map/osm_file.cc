#include "osm_file.h"
#include <boost/format.hpp>
#include <cstdint>
#include <iostream>

namespace lanelet {
namespace osm {
namespace {
struct UnresolvedRole {
  Id relation{};
  Id referencedRelation{};
  Primitive** location{};
};

Attributes tags(const tinyxml2::XMLElement* node) {
  Attributes attributes;
  for (auto* tag = node->FirstChildElement(keyword::Tag); tag != nullptr;
       tag = tag->NextSiblingElement(keyword::Tag)) {
    const char* key = tag->Attribute(keyword::Key);
    const char* value = tag->Attribute(keyword::Value);
    if (key == nullptr || value == nullptr) {
      continue;
    }
    if (std::string(key) == keyword::Elevation) {
      continue;
    }
    attributes[key] = value;
  }
  return attributes;
}

bool isDeleted(const tinyxml2::XMLElement* node) {
  const char* action = node->Attribute(keyword::Action);
  return action != nullptr && std::string(action) == keyword::Delete;
}

std::string toJosmStyle(double d) {
  std::string str = boost::str(boost::format{"%12.11f"} % d);
  str.erase(str.find_last_not_of('0') + 1, std::string::npos);
  str.erase(str.find_last_not_of('.') + 1, std::string::npos);
  return str;
}

void removeAndFixPlaceholders(Primitive** toRemove, Roles& fromRoles, std::vector<UnresolvedRole>& placeholders) {
  // find other placeholders that we have to fix
  auto remIt =
      std::find_if(fromRoles.begin(), fromRoles.end(), [&](const Role& role) { return &role.second == toRemove; });
  assert(remIt != fromRoles.end());
  std::vector<std::pair<size_t, Primitive**>> placeholderLocations;
  for (auto it = fromRoles.begin(); it != fromRoles.end(); ++it) {
    if (it->second == nullptr && remIt != it) {
      auto idx = std::distance(fromRoles.begin(), it);
      placeholderLocations.emplace_back(it > remIt ? idx - 1 : idx, &it->second);
    }
  }
  fromRoles.erase(remIt);
  if (placeholderLocations.empty()) {
    return;  // nothing to update
  }
  // get the new locations
  std::map<Primitive**, Primitive**> newLocations;
  for (auto& loc : placeholderLocations) {
    newLocations.emplace(loc.second, &std::next(fromRoles.begin(), long(loc.first))->second);
  }
  // adapt existing locations
  for (auto& placeholder : placeholders) {
    auto it = newLocations.find(placeholder.location);
    if (it != newLocations.end()) {
      placeholder.location = it->second;
    }
  }
}

class OsmFileWriter {
 public:
  static std::unique_ptr<tinyxml2::XMLDocument> write(const File& osmFile) {
    auto xml = std::make_unique<tinyxml2::XMLDocument>();
    auto* osmNode = xml->NewElement(keyword::Osm);
    xml->InsertEndChild(osmNode);
    osmNode->SetAttribute("version", "0.6");
    osmNode->SetAttribute("generator", "lanelet2");
    lanelet::osm::OsmFileWriter::writeNodes(osmNode, osmFile.nodes);
    lanelet::osm::OsmFileWriter::writeWays(osmNode, osmFile.ways);
    lanelet::osm::OsmFileWriter::writeRelations(osmNode, osmFile.relations);
    return xml;
  }

 private:
  static void writeAttributes(tinyxml2::XMLElement* elemNode,
                              const Attributes& attributes) {
    for (const auto& attribute : attributes) {
      auto* tagNode = elemNode->GetDocument()->NewElement(keyword::Tag);
      elemNode->InsertEndChild(tagNode);
      tagNode->SetAttribute(keyword::Key, attribute.first.c_str());
      tagNode->SetAttribute(keyword::Value, attribute.second.c_str());
    }
  }

  static void writeNodes(tinyxml2::XMLElement* osmNode, const Nodes& nodes) {
    for (const auto& node : nodes) {
      auto* xmlNode = osmNode->GetDocument()->NewElement(keyword::Node);
      osmNode->InsertEndChild(xmlNode);
      xmlNode->SetAttribute(keyword::Id,
                             static_cast<std::int64_t>(node.second.id));
      if (node.second.id > 0) {
        xmlNode->SetAttribute(keyword::Visible, "true");
        xmlNode->SetAttribute(keyword::Version, 1);
      }
      xmlNode->SetAttribute(keyword::Lat, toJosmStyle(node.second.point.lat).c_str());
      xmlNode->SetAttribute(keyword::Lon, toJosmStyle(node.second.point.lon).c_str());

      if (node.second.point.ele != 0.) {
        auto* tagNode = xmlNode->GetDocument()->NewElement(keyword::Tag);
        xmlNode->InsertEndChild(tagNode);
        tagNode->SetAttribute(keyword::Key, keyword::Elevation);
        tagNode->SetAttribute(keyword::Value, node.second.point.ele);
      }
      writeAttributes(xmlNode, node.second.attributes);
    }
  }

  static void writeWays(tinyxml2::XMLElement* osmNode, const Ways& ways) {
    for (const auto& wayElem : ways) {
      const auto& way = wayElem.second;
      auto* xmlNode = osmNode->GetDocument()->NewElement(keyword::Way);
      osmNode->InsertEndChild(xmlNode);
      xmlNode->SetAttribute(keyword::Id, static_cast<std::int64_t>(way.id));
      if (way.id > 0) {
        xmlNode->SetAttribute(keyword::Visible, "true");
        xmlNode->SetAttribute(keyword::Version, 1);
      }
      for (const auto& node : way.nodes) {
        auto* nd = xmlNode->GetDocument()->NewElement(keyword::Nd);
        xmlNode->InsertEndChild(nd);
        nd->SetAttribute(keyword::Ref, static_cast<std::int64_t>(node->id));
      }
      writeAttributes(xmlNode, way.attributes);
    }
  }

  static void writeRelations(tinyxml2::XMLElement* osmNode,
                             const Relations& relations) {
    for (const auto& relationElem : relations) {
      const auto& relation = relationElem.second;
      auto* xmlNode = osmNode->GetDocument()->NewElement(keyword::Relation);
      osmNode->InsertEndChild(xmlNode);
      xmlNode->SetAttribute(keyword::Id,
                             static_cast<std::int64_t>(relation.id));
      if (relation.id > 0) {
        xmlNode->SetAttribute(keyword::Visible, "true");
        xmlNode->SetAttribute(keyword::Version, 1);
      }
      for (const auto& role : relation.members) {
        auto* xmlMember = xmlNode->GetDocument()->NewElement(keyword::Member);
        xmlNode->InsertEndChild(xmlMember);
        auto type = role.second->type();
        xmlMember->SetAttribute(keyword::Type, type.c_str());
        xmlMember->SetAttribute(keyword::Ref,
                                static_cast<std::int64_t>(role.second->id));
        xmlMember->SetAttribute(keyword::Role, role.first.c_str());
      }
      writeAttributes(xmlNode, relation.attributes);
    }
  }
};

class OsmFileParser {
 public:
  static File read(const tinyxml2::XMLDocument& document,
                   Errors* errors = nullptr) {
    OsmFileParser osmParser;
    File file;
    const auto* osmNode = document.FirstChildElement(keyword::Osm);
    if (osmNode == nullptr) {
      if (errors != nullptr) {
        errors->emplace_back("OSM document has no osm root element");
      }
      return file;
    }
    file.nodes = lanelet::osm::OsmFileParser::readNodes(osmNode);
    file.ways = osmParser.readWays(osmNode, file.nodes, file.waysIds);
    file.relations = osmParser.readRelations(osmNode, file.nodes, file.ways);
    if (errors != nullptr) {
      *errors = osmParser.errors_;
    }
    return file;
  }

 private:
  static Nodes readNodes(const tinyxml2::XMLElement* osmNode) {
    Nodes nodes;
    for (auto* node = osmNode->FirstChildElement(keyword::Node); node != nullptr;
         node = node->NextSiblingElement(keyword::Node)) {
      if (isDeleted(node)) {
        continue;
      }
      const auto id = node->Int64Attribute(keyword::Id, InvalId);
      const auto attributes = tags(node);
      const auto lat = node->DoubleAttribute(keyword::Lat, 0.);
      const auto lon = node->DoubleAttribute(keyword::Lon, 0.);
      double ele = 0.;
      for (auto* tag = node->FirstChildElement(keyword::Tag); tag != nullptr;
           tag = tag->NextSiblingElement(keyword::Tag)) {
        const char* key = tag->Attribute(keyword::Key);
        if (key != nullptr && std::string(key) == keyword::Elevation) {
          ele = tag->DoubleAttribute(keyword::Value, 0.);
          break;
        }
      }
      nodes[id] = Node{id, attributes, {lat, lon, ele}};
    }
    return nodes;
  }

  Ways readWays(const tinyxml2::XMLElement* osmNode, Nodes& nodes,
                IdSet& waysIds) {
    Ways ways;
    for (auto* node = osmNode->FirstChildElement(keyword::Way); node != nullptr;
         node = node->NextSiblingElement(keyword::Way)) {
      if (isDeleted(node)) {
        continue;
      }
      const auto id = node->Int64Attribute(keyword::Id, InvalId);
      const auto attributes = tags(node);
      const auto nodeIds = [node] {
        Ids ids;
        for (auto* refNode = node->FirstChildElement(keyword::Nd);
             refNode != nullptr;
             refNode = refNode->NextSiblingElement(keyword::Nd)) {
          ids.push_back(refNode->Int64Attribute(keyword::Ref));
        }
        return ids;
      }();
      std::vector<Node*> wayNodes;
      for (const auto& nodeId : nodeIds) {
        auto it = nodes.find(nodeId);
        if (it != nodes.end()) {
          wayNodes.push_back(&(it->second));
        } else {
          reportParseError(id, "Way references nonexisting point");
          wayNodes.clear();
          break;
        }
      }
      ways[id] = Way{id, attributes, wayNodes};
      waysIds.insert(id);
    }
    return ways;
  }

  Relations readRelations(const tinyxml2::XMLElement* osmNode, Nodes& nodes,
                          Ways& ways) {
    Relations relations;
    // Two-pass approach: We can resolve all roles except where relations reference relations. We insert a dummy nullptr
    // and resolve that later on.
    std::vector<UnresolvedRole> unresolvedRoles;
    for (auto* node = osmNode->FirstChildElement(keyword::Relation);
         node != nullptr; node = node->NextSiblingElement(keyword::Relation)) {
      if (isDeleted(node)) {
        continue;
      }
      const auto id = node->Int64Attribute(keyword::Id, InvalId);
      const auto attributes = tags(node);
      auto& relation = relations.emplace(id, Relation{id, attributes, {}}).first->second;

      // resolve members
      auto& roles = relation.members;
      for (auto* member = node->FirstChildElement(keyword::Member);
           member != nullptr;
           member = member->NextSiblingElement(keyword::Member)) {
        Id memberId = member->Int64Attribute(keyword::Ref);
        const std::string role = member->Attribute(keyword::Role) == nullptr
                                     ? ""
                                     : member->Attribute(keyword::Role);
        const std::string type = member->Attribute(keyword::Type) == nullptr
                                     ? ""
                                     : member->Attribute(keyword::Type);
        try {
          if (type == keyword::Node) {
            roles.emplace_back(role, &nodes.at(memberId));
          } else if (type == keyword::Way) {
            roles.emplace_back(role, &ways.at(memberId));
          } else if (type == keyword::Relation) {
            // insert a placeholder and store a pointer to it for the second pass
            roles.emplace_back(role, nullptr);
            unresolvedRoles.push_back(UnresolvedRole{id, memberId, &roles.back().second});
          }
        } catch (std::out_of_range&) {
          reportParseError(id, "Relation has nonexistent member " + std::to_string(memberId));
        }
      }
    }

    // now resolve all unresolved roles that point to other relations
    for (const auto& unresolvedRole : unresolvedRoles) {
      try {
        assert(*unresolvedRole.location == nullptr);
        *unresolvedRole.location = &relations.at(unresolvedRole.referencedRelation);
      } catch (std::out_of_range&) {
        reportParseError(unresolvedRole.relation, "Relation references nonexistent relation " +
                                                      std::to_string(unresolvedRole.referencedRelation));
        // now it gets ugly: find placeholder and remove it. Fix all other placeholders because the pointers are
        // invalidated after moving. This is inefficent, but its the fault of the guy loading an invalid map, not ours.
        auto& relation = relations.at(unresolvedRole.relation);
        removeAndFixPlaceholders(unresolvedRole.location, relation.members, unresolvedRoles);
      }
    }
    return relations;
  }

  OsmFileParser() = default;
  void reportParseError(Id id, const std::string& what) {
    auto errstr = "Error reading primitive with id " + std::to_string(id) + " from file: " + what;
    errors_.push_back(errstr);
  }
  Errors errors_;
};
}  // namespace

bool operator==(const Way& lhs, const Way& rhs) {
  auto nodesEqual = [&lhs, &rhs]() {
    for (auto i = 0u; i < lhs.nodes.size(); i++) {
      if (lhs.nodes[i]->id != rhs.nodes[i]->id) {
        return false;
      }
    }
    return true;
  };
  return lhs.id == rhs.id && lhs.nodes.size() == rhs.nodes.size() && nodesEqual();
}

bool operator==(const Relation& lhs, const Relation& rhs) {
  auto membersEqual = [&lhs, &rhs]() {
    for (auto itL = lhs.members.begin(), itR = rhs.members.begin(); itL != lhs.members.end(); ++itL, ++itR) {
      if (itL->second->type() != itR->second->type() && itL->second->id != itR->second->id) {
        return false;
      }
    }
    return true;
  };
  return lhs.id == rhs.id && lhs.members.size() == rhs.members.size() && membersEqual();
}

bool operator==(const File& lhs, const File& rhs) {
  return lhs.nodes == rhs.nodes && lhs.ways == rhs.ways && lhs.relations == rhs.relations;
}

File read(const tinyxml2::XMLDocument& document, Errors* errors) {
  return OsmFileParser::read(document, errors);
}

std::unique_ptr<tinyxml2::XMLDocument> write(const File& file) {
  return OsmFileWriter::write(file);
}
}  // namespace osm
}  // namespace lanelet
