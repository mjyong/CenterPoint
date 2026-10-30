"""The online node's topics of a rosbag2 (sqlite3), up to the first N clouds: a bag small enough for the
Orin (``ros2 bag play``, dogperc_bag_replay).

Rows are copied as stored (CDR bytes, receive times, the recorder's QoS) into the source's own schema, and
metadata.yaml keeps the source's version, so ROS 2 Humble reads the cut as it reads the source (rosbags 0.11
only writes versions 8 / 9). The messages are the ones ``cpp_parity.py python --bag <source> --max-clouds N``
replays: everything before cloud N + 1.

    python tools/deploy/cut_bag.py --bag /zjst/robot/rosbag0921/rosbag2_2026_09_21-16_41_27 --robot kygi \\
        --max-clouds 600 --out data/deploy/T30/rosbag2_0921_600
"""
import argparse
import copy
import heapq
import os
import sqlite3
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


class _Dumper(yaml.SafeDumper):
    pass


# multi-line strings (offered_qos_profiles) as literal blocks, as rosbag2 writes them
_Dumper.add_representer(
    str, lambda d, v: d.represent_scalar("tag:yaml.org,2002:str", v, style="|" if "\n" in v else None))


def _rows(db, topics):
    q = ("SELECT m.timestamp, t.name, m.data FROM messages m JOIN topics t ON m.topic_id = t.id "
         "WHERE t.name IN (%s) ORDER BY m.timestamp, m.id" % ",".join("?" * len(topics)))
    yield from db.execute(q, list(topics))


def cut(bag, out, topics, lidar_topic, max_clouds):
    """Returns {topic: messages written}."""
    with open(os.path.join(bag, "metadata.yaml"), encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    info = meta["rosbag2_bagfile_information"]
    if info.get("storage_identifier") != "sqlite3" or info.get("compression_mode"):
        raise SystemExit("%s: only uncompressed sqlite3 bags" % bag)
    srcs = [sqlite3.connect("file:%s?mode=ro" % os.path.join(bag, p), uri=True) for p in info["relative_file_paths"]]
    os.makedirs(out)
    name = os.path.basename(os.path.normpath(out)) + "_0.db3"
    dst = sqlite3.connect(os.path.join(out, name))
    for (sql,) in srcs[0].execute("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL ORDER BY rowid"):
        dst.execute(sql)
    if dst.execute("SELECT count(*) FROM sqlite_master WHERE name = 'schema'").fetchone()[0]:
        dst.executemany("INSERT INTO schema VALUES (?, ?)", srcs[0].execute("SELECT * FROM schema").fetchall())
    ids = {}
    for db in srcs:
        for row in db.execute("SELECT name, type, serialization_format, offered_qos_profiles FROM topics"):
            if row[0] in topics and row[0] not in ids:
                ids[row[0]] = len(ids) + 1
                dst.execute("INSERT INTO topics (id, name, type, serialization_format, offered_qos_profiles) "
                            "VALUES (?, ?, ?, ?, ?)", (ids[row[0]],) + tuple(row))
    missing = set(topics) - set(ids)
    if missing:
        raise SystemExit("%s: no messages on %s" % (bag, ", ".join(sorted(missing))))
    counts, clouds, first, last = dict.fromkeys(ids, 0), 0, None, None
    for t, topic, data in heapq.merge(*(_rows(db, topics) for db in srcs), key=lambda r: r[0]):
        if topic == lidar_topic:
            if clouds == max_clouds:
                break
            clouds += 1
        dst.execute("INSERT INTO messages (topic_id, timestamp, data) VALUES (?, ?, ?)", (ids[topic], t, data))
        counts[topic] += 1
        first = t if first is None else first
        last = t
    dst.commit()
    dst.close()

    total, span = sum(counts.values()), (last - first) if first is not None else 0
    info = copy.deepcopy(info)
    info.update(duration=dict(nanoseconds=span), starting_time=dict(nanoseconds_since_epoch=first or 0),
                message_count=total, relative_file_paths=[name],
                topics_with_message_count=[dict(e, message_count=counts[e["topic_metadata"]["name"]])
                                           for e in info["topics_with_message_count"]
                                           if e["topic_metadata"]["name"] in counts])
    if "files" in info:
        info["files"] = [dict(path=name, starting_time=dict(nanoseconds_since_epoch=first or 0),
                              duration=dict(nanoseconds=span), message_count=total)]
    with open(os.path.join(out, "metadata.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(dict(rosbag2_bagfile_information=info), f, Dumper=_Dumper, sort_keys=False)
    return counts


def main():
    from dog_perception.node_config import node_parameters
    from message_log import node_topics

    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--robot", default="kygi", help="node preset: its lidar, IMU and pose topics are kept")
    ap.add_argument("--max-clouds", type=int, required=True)
    ap.add_argument("--out", required=True, help="new bag directory (must not exist)")
    args = ap.parse_args()
    params = node_parameters(args.robot)
    counts = cut(args.bag, args.out, node_topics(params), params["lidar_topic"], args.max_clouds)
    print("%s: %s" % (args.out, ", ".join("%d %s" % (n, t) for t, n in counts.items())))


if __name__ == "__main__":
    main()
