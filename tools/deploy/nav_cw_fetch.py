"""Download what dog 8 lacks for building our navigation stack (T55 lists it), on a machine with a good connection
(the computer next to the dog), check every file's SHA256, and leave a folder to copy to the dog. Standard library
only: runs on Linux, macOS and Windows with Python 3.

    python3 nav_cw_fetch.py fetch_urls.txt nav_cw_downloads \
        [--rewrite http://ports.ubuntu.com/ubuntu-ports=https://mirrors.tuna.tsinghua.edu.cn/ubuntu-ports]
    scp -r nav_cw_downloads nvidia@<dog>:/home/nvidia/          (then on the dog: sha256sum -c SHA256SUMS)

fetch_urls.txt, one file per line ('#' starts a comment):
    URL [sha256=HEX] [name=FILE]
or what `apt-get install --print-uris` prints:
    'URL' FILE SIZE SHA256:HEX
--rewrite FROM=TO (repeatable) swaps a URL prefix for a mirror; the hash still has to match. A file already in the
folder with the right hash is not downloaded again. Exit 1 if any file failed.
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request


def parse(lines):
    """-> [(url, name, sha256 or None)]"""
    out = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("'"):                      # apt-get --print-uris
            parts = line.split()
            url = parts[0].strip("'")
            name = parts[1] if len(parts) > 1 else None
            sha = next((p.split(":", 1)[1] for p in parts[2:] if p.upper().startswith("SHA256:")), None)
        else:
            parts = line.split()
            url, sha, name = parts[0], None, None
            for p in parts[1:]:
                if p.startswith("sha256="):
                    sha = p.split("=", 1)[1]
                elif p.startswith("name="):
                    name = p.split("=", 1)[1]
        if not name:
            name = os.path.basename(urllib.parse.urlparse(url).path)
        out.append((url, name, sha.lower() if sha else None))
    names = [n for _, n, _ in out]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise SystemExit("the same file name twice (give name=...): %s" % ", ".join(dup))
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rewrite(url, rules):
    for src, dst in rules:
        if url.startswith(src):
            return dst + url[len(src):]
    return url


def fetch(url, path, tries=3, timeout=60):
    tmp = path + ".part"
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nav-cw-fetch"})
            with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(tmp, path)
            return None
        except Exception as e:  # noqa: BLE001  (network errors of every kind: retry, then report)
            err = "%s: %s" % (type(e).__name__, e)
            if os.path.exists(tmp):
                os.remove(tmp)
            time.sleep(min(2 ** k, 10))
    return err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("list")
    ap.add_argument("out")
    ap.add_argument("--rewrite", action="append", default=[], metavar="FROM=TO")
    ap.add_argument("--tries", type=int, default=3)
    args = ap.parse_args(argv)
    rules = []
    for r in args.rewrite:
        if "=" not in r:
            raise SystemExit("--rewrite wants FROM=TO: %s" % r)
        rules.append(tuple(r.split("=", 1)))
    with open(args.list, encoding="utf-8") as f:
        items = parse(f)
    os.makedirs(args.out, exist_ok=True)
    failed, sums, total = [], [], 0
    for i, (url, name, want) in enumerate(items, 1):
        path = os.path.join(args.out, name)
        src = rewrite(url, rules)
        if os.path.exists(path) and want and sha256(path) == want:
            state = "have"
        else:
            err = fetch(src, path, tries=args.tries)
            if err:
                failed.append((name, err))
                print("[%d/%d] FAILED %s: %s" % (i, len(items), name, err))
                continue
            state = "got"
        got = sha256(path)
        if want and got != want:
            failed.append((name, "sha256 %s, expected %s" % (got, want)))
            print("[%d/%d] BAD HASH %s (from %s)" % (i, len(items), name, src))
            os.remove(path)
            continue
        size = os.path.getsize(path)
        total += size
        sums.append("%s  %s" % (got, name))
        print("[%d/%d] %-4s %s (%.1f MB)%s" % (i, len(items), state, name, size / 1e6, "" if want else "  no hash given"))
    with open(os.path.join(args.out, "SHA256SUMS"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(sums) + ("\n" if sums else ""))
    print("%d of %d files, %.1f MB in %s; SHA256SUMS written" % (len(sums), len(items), total / 1e6, args.out))
    for name, err in failed:
        print("  failed: %s: %s" % (name, err))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
