"""Convert a ros2 bag recorded on dog 8 to mcap for Foxglove and the planning tools, keeping every topic.

The dog's ros2 bag has no mcap plugin, so it records sqlite3 (T61); a plain conversion drops the topics whose types it
does not know (kygi665_msgs/INSPVAX, perception_interface/LidarObjectArray, mc_interface/Motion, rhr_interface ...).
Here the definitions come from the repository (bag_on_map.typestore, every package under Nav_D20/seres_interface/ros),
and the messages are copied as they are (CDR, not decoded and encoded again). Topics whose type is still unknown are
listed, not silently dropped.

    python Nav_D20/planning/tools/bag_to_mcap.py <bag dir> <out dir>
"""
import argparse
import os
import sys
from collections import Counter
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def convert(src, dst):
    """-> (Counter of messages written per topic, {topic: why it was left out})"""
    from bag_on_map import typestore
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag2 import StoragePlugin, Writer

    ts = typestore(all_packages=True, quiet=True)
    written, left_out = Counter(), {}
    with AnyReader([Path(src)], default_typestore=ts) as reader, \
            Writer(Path(dst), version=8, storage_plugin=StoragePlugin.MCAP) as writer:
        out = {}
        for c in reader.connections:
            if c.msgtype not in ts.types:
                left_out[c.topic] = "type %s unknown" % c.msgtype
                continue
            key = (c.topic, c.msgtype)
            if key not in out:
                out[key] = writer.add_connection(c.topic, c.msgtype, typestore=ts)
        conns = [c for c in reader.connections if (c.topic, c.msgtype) in out]
        for c, t, raw in reader.messages(connections=conns):
            writer.write(out[(c.topic, c.msgtype)], t, raw)
            written[c.topic] += 1
    return written, left_out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bag", help="a ros2 bag directory (sqlite3 or mcap)")
    ap.add_argument("out", help="the mcap bag directory to write (must not exist)")
    args = ap.parse_args(argv)
    written, left_out = convert(args.bag, args.out)
    for topic in sorted(written):
        print("  %-50s %8d" % (topic, written[topic]))
    for topic, why in sorted(left_out.items()):
        print("  LEFT OUT %s: %s" % (topic, why))
    print("%s: %d topics, %d messages%s" % (args.out, len(written), sum(written.values()),
                                           ", %d topics left out" % len(left_out) if left_out else ""))
    return 1 if left_out else 0


if __name__ == "__main__":
    sys.exit(main())
