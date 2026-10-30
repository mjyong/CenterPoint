"""Is foxglove_bridge serving what Foxglove will show? Connects like Foxglove does (WebSocket, subprotocol
foxglove.websocket.v1), lists the advertised channels, subscribes to some topics for a few seconds and counts
their messages. Standard library only, so it runs on the board (deploy_deb's scripts/orin_foxglove.sh).

    python3 tools/deploy/foxglove_probe.py --url ws://localhost:8765 --seconds 5 \\
        --topics /perception/lidar_3d/tracks /perception/lidar_3d/predictions /perception/lidar_3d/cloud /tf

Exit 1 when a topic is not advertised or sent nothing.
"""
import argparse
import base64
import json
import os
import socket
import struct
import sys
import time
from urllib.parse import urlparse

PROTOCOL = "foxglove.websocket.v1"


class WebSocket:
    def __init__(self, url, timeout=5.0):
        u = urlparse(url)
        self.sock = socket.create_connection((u.hostname, u.port or 80), timeout=timeout)
        key = base64.b64encode(os.urandom(16)).decode()
        self.sock.sendall(("GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
                           "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Protocol: %s\r\n\r\n"
                           % (u.path or "/", u.hostname, u.port or 80, key, PROTOCOL)).encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = self.sock.recv(1)
            if not chunk:
                raise ConnectionError("closed during the handshake")
            head += chunk
        if b" 101 " not in head.split(b"\r\n", 1)[0]:
            raise ConnectionError("no WebSocket upgrade: %s" % head.split(b"\r\n", 1)[0].decode(errors="replace"))

    def _read(self, n):
        out = b""
        while len(out) < n:
            chunk = self.sock.recv(n - len(out))
            if not chunk:
                raise ConnectionError("closed")
            out += chunk
        return out

    def send(self, payload, opcode=0x1):
        """A masked frame (client to server)."""
        data = payload.encode() if isinstance(payload, str) else payload
        n = len(data)
        head = bytes([0x80 | opcode])
        if n < 126:
            head += bytes([0x80 | n])
        elif n < 1 << 16:
            head += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            head += bytes([0x80 | 127]) + struct.pack(">Q", n)
        mask = os.urandom(4)
        self.sock.sendall(head + mask + bytes(b ^ mask[i % 4] for i, b in enumerate(data)))

    def recv(self):
        """(opcode, payload) of the next whole message; pings are answered."""
        parts, first = [], None
        while True:
            b0, b1 = self._read(2)
            n = b1 & 0x7F
            if n == 126:
                n = struct.unpack(">H", self._read(2))[0]
            elif n == 127:
                n = struct.unpack(">Q", self._read(8))[0]
            mask = self._read(4) if b1 & 0x80 else None
            data = self._read(n)
            if mask:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            op = b0 & 0x0F
            if op == 0x9:
                self.send(data, 0xA)
                continue
            if op in (0x1, 0x2, 0x8):
                first = op
            parts.append(data)
            if b0 & 0x80:
                return first, b"".join(parts)


def probe(url, topics, seconds):
    ws = WebSocket(url)
    server, channels, counts = None, {}, {}
    subscribed = {}
    end = time.time() + seconds
    sub_id = 0
    while time.time() < end:
        ws.sock.settimeout(max(0.1, end - time.time()))
        try:
            op, data = ws.recv()
        except (socket.timeout, ConnectionError):
            break
        if op == 0x8:
            break
        if op == 0x1:
            msg = json.loads(data)
            if msg.get("op") == "serverInfo":
                server = dict(name=msg.get("name"), capabilities=msg.get("capabilities", []))
            elif msg.get("op") == "advertise":
                for c in msg.get("channels", []):
                    channels[c["topic"]] = dict(id=c["id"], schema=c.get("schemaName"), encoding=c.get("encoding"))
                    if c["topic"] in topics and c["topic"] not in subscribed.values():
                        sub_id += 1
                        subscribed[sub_id] = c["topic"]
                        ws.send(json.dumps(dict(op="subscribe", subscriptions=[dict(id=sub_id, channelId=c["id"])])))
        elif op == 0x2 and data[:1] == b"\x01" and len(data) >= 5:      # message data: opcode, subscription id
            topic = subscribed.get(struct.unpack("<I", data[1:5])[0])
            if topic:
                counts[topic] = counts.get(topic, 0) + 1
    ws.sock.close()
    return dict(server=server, channels={t: c["schema"] for t, c in sorted(channels.items())},
                received={t: counts.get(t, 0) for t in topics}, seconds=seconds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765")
    ap.add_argument("--topics", nargs="*", default=["/perception/lidar_3d/tracks", "/perception/lidar_3d/predictions",
                                                    "/perception/lidar_3d/cloud", "/tf"],
                    help="default: what deploy_deb's Foxglove layout shows")
    ap.add_argument("--seconds", type=float, default=5.0)
    args = ap.parse_args()
    report = probe(args.url, args.topics, args.seconds)
    missing = [t for t in args.topics if t not in report["channels"]]
    silent = [t for t in args.topics if t in report["channels"] and not report["received"][t]]
    report.update(missing=missing, silent=silent, passed=not missing and not silent)
    print(json.dumps(report, indent=1))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
