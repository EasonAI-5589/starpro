#!/usr/bin/env bash
set -euo pipefail

# Scan the current contents of every tracked file, including this checker.
# Pattern fragments below keep the rule definitions from matching themselves.
python3 - <<'PY'
import ipaddress
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys


def git(*args):
    result = subprocess.run(
        ["git", *args], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    if result.returncode:
        raise SystemExit("ERROR: could not enumerate tracked files")
    return result.stdout


root = Path(os.fsdecode(git("rev-parse", "--show-toplevel").rstrip(b"\n")))
os.chdir(root)
paths = sorted(set(git("ls-files", "-z").split(b"\0")) - {b""})

platform = re.compile("ai" + "hc" + "|" + "\u767e\u8238", re.IGNORECASE)
credentials = re.compile(
    r"(?:AKI[A]|ASI[A])[0-9A-Z]{16}"
    r"|gh[pousr]_[A-Za-z0-9_]{20,}"
    r"|github[_]pat_[A-Za-z0-9_]{20,}"
    r"|s[k]-[A-Za-z0-9_-]{20,}"
    r"|h[f]_[A-Za-z0-9]{20,}"
    r"|xox[baprs]-[A-Za-z0-9-]{16,}"
    r"|-----BEGIN (?:RSA |OPENSSH |EC |DSA |ENCRYPTED )?PRIVATE KEY-----"
    r"|(?:https?|socks5?)://[^\s/:@]+:[^\s/@]+@",
    re.IGNORECASE,
)
literal_secret = re.compile(
    r"(?im)\b(?:api[_-]?key|access[_-]?key[_-]?secret|secret[_-]?key|"
    r"client[_-]?secret|password|passwd|(?:[a-z]+[_-])?token)\b"
    r"[\"']?\s*[:=]\s*[\"']([A-Za-z0-9_+/.=-]{16,})[\"']"
)
placeholder = re.compile(
    r"(?i)^(?:your[_-]|example|placeholder|replace[_-]|dummy|test[_-]|"
    r"changeme|[x*]+$|[A-Z][A-Z0-9_]*_HERE$)"
)
internal_path = re.compile(
    r"/(?:mnt|home|Users|root)/(?:[A-Za-z0-9_.-]+)(?:/|\b)"
    r"|[A-Z]:[\\/](?:Users|mnt)[\\/]",
    re.IGNORECASE,
)
private_host = re.compile(
    r"\bcce[-_]pmm\b|\brdma[/]hca\b|\b[a-z0-9.-]+\.(?:internal|local)\b",
    re.IGNORECASE,
)
ipv4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
private_networks = tuple(
    ipaddress.ip_network(prefix)
    for prefix in ("10" + ".0.0.0/8", "172" + ".16.0.0/12", "192" + ".168.0.0/16")
)
scheduler = re.compile(
    r"(?im)^\s*(?:#SBATCH\b|#PBS\b|#BSUB\b|"
    r"kind:\s*(?:Job|CronJob|PyTorchJob|MPIJob|TFJob)\b|"
    r"apiVersion:\s*(?:batch|kubeflow|scheduling\.[a-z0-9.-]+)/)"
    r"|\b(?:sharedMemory|resourcePoolId|queueId|clusterId)\s*[\"']?\s*:",
)
artifact_path = re.compile(
    r"(?:^|/)(?:answers(?:_upload)?|results(?:[_-][^/]*)?|outputs?|logs?|"
    r"eval_submissions|rebuttal|wandb|runs|checkpoints?|__pycache__|"
    r"\.cache|\.venv|venv)(?:/|$)"
    r"|\.(?:jsonl|xlsx?|tsv|log|out|output|py[co]|bin|pt|pth|ckpt|"
    r"safetensors|gguf|onnx|h5|pkl|npy|npz|tar(?:\.gz)?|tgz|zip|7z|pem|key)$"
    r"|(?:^|/)(?:\.env(?:\.(?!example$)[^/]+)?|id_(?:rsa|ed25519)[^/]*|"
    r"bot\d*_monitor_status\.md|\.DS_Store)$"
    r"|\.(?:pre_[^/]*|bak(?:_[^/]*)?)$",
    re.IGNORECASE,
)


def secret_values(text):
    return [
        match for match in literal_secret.finditer(text)
        if not placeholder.match(match.group(1))
    ]


def categories(text):
    found = set()
    for label, pattern in (
        ("private platform", platform),
        ("credential-like material", credentials),
        ("private absolute path", internal_path),
        ("private infrastructure", private_host),
        ("scheduler configuration", scheduler),
    ):
        if pattern.search(text):
            found.add(label)
    if secret_values(text):
        found.add("credential-like material")
    for match in ipv4.finditer(text):
        try:
            address = ipaddress.ip_address(match.group())
        except ValueError:
            continue
        if any(address in network for network in private_networks):
            found.add("private network address")
    return found


def safe_label(name):
    # Filenames may themselves contain secrets or terminal control characters.
    name = credentials.sub("<redacted>", name)
    for match in reversed(secret_values(name)):
        name = name[:match.start(1)] + "<redacted>" + name[match.end(1):]
    return json.dumps(name, ensure_ascii=True)


failed = False
for raw_path in paths:
    name = os.fsdecode(raw_path)
    path = Path(name)
    found = categories(name)
    if artifact_path.search(name):
        found.add("generated or private artifact")
    try:
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            # Inspect the link without reading any target outside the checkout.
            contents = os.readlink(path)
            target = (path.parent / contents).resolve()
            if os.path.isabs(contents) or root not in target.parents:
                found.add("external symlink")
        elif stat.S_ISREG(mode):
            contents = path.read_bytes().decode("utf-8", errors="replace")
        else:
            contents = ""
            found.add("unsupported tracked entry")
        found.update(categories(contents))
    except FileNotFoundError:
        found.add("missing tracked entry; stage its deletion first")
    except (OSError, RuntimeError):
        found.add("unreadable tracked entry")
    if found:
        failed = True
        print(f"ERROR: {safe_label(name)}: {', '.join(sorted(found))}", file=sys.stderr)

if failed:
    raise SystemExit(1)
print(f"PUBLIC_RELEASE_CHECK_OK tracked_files={len(paths)}")
PY
