#!/usr/bin/env python3
"""Record control-plane and data-plane access to the live GB10 host."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run(cmd: list[str], timeout: int = 20) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return {
            "cmd": cmd,
            "returncode": proc.returncode,
            "elapsed_s": time.perf_counter() - started,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - diagnostic path
        return {
            "cmd": cmd,
            "elapsed_s": time.perf_counter() - started,
            "error": repr(exc),
        }


def scrub_result(result: dict[str, Any], *, scrub_stdout: bool) -> dict[str, Any]:
    public = dict(result)
    if scrub_stdout and public.get("stdout"):
        public["stdout"] = "<redacted; target peer summary recorded separately>"
    return public


def tcp_probe(host: str, port: int, timeout_s: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return {
                "host": host,
                "port": port,
                "ok": True,
                "elapsed_s": time.perf_counter() - started,
            }
    except Exception as exc:
        return {
            "host": host,
            "port": port,
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error": repr(exc),
        }


def parse_tailscale_peer(status_json: dict[str, Any], host: str) -> dict[str, Any] | None:
    peers = status_json.get("Peer")
    if not isinstance(peers, dict):
        return None
    for peer in peers.values():
        if not isinstance(peer, dict):
            continue
        tailscale_ips = peer.get("TailscaleIPs") or []
        dns_name = str(peer.get("DNSName") or "")
        host_name = str(peer.get("HostName") or "")
        if host in tailscale_ips or host in dns_name or host in host_name:
            return {
                "dns_name": peer.get("DNSName"),
                "host_name": peer.get("HostName"),
                "os": peer.get("OS"),
                "online": peer.get("Online"),
                "active": peer.get("Active"),
                "tailscale_ips": tailscale_ips,
                "relay": peer.get("Relay"),
                "last_seen": peer.get("LastSeen"),
                "rx_bytes": peer.get("RxBytes"),
                "tx_bytes": peer.get("TxBytes"),
                "cur_addr": peer.get("CurAddr"),
            }
    return None


def load_status_json(tailscale: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    result = run([tailscale, "status", "--json"], timeout=20)
    if result.get("returncode") != 0 or not result.get("stdout"):
        return None, result
    try:
        return json.loads(result["stdout"]), result
    except json.JSONDecodeError as exc:
        result["json_error"] = str(exc)
        return None, result


def collect(args: argparse.Namespace) -> dict[str, Any]:
    tailscale = shutil.which(args.tailscale_bin)
    report: dict[str, Any] = {
        "schema": "gb10-host-access-probe/v1",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "target": {
            "host": args.host,
            "name": args.name,
            "ssh_user": args.ssh_user,
            "ssh_port": args.ssh_port,
        },
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": sys.version,
            "executable": sys.executable,
        },
        "tailscale": {"available": bool(tailscale), "path": tailscale},
        "commands": {},
        "tcp": {},
        "peer": None,
        "findings": [],
    }

    if tailscale:
        status_json, status_result = load_status_json(tailscale)
        report["commands"]["tailscale_status_json"] = scrub_result(
            status_result,
            scrub_stdout=not args.include_raw_tailscale_status,
        )
        report["peer"] = (
            parse_tailscale_peer(status_json, args.host) if status_json else None
        )
        status_text = run([tailscale, "status"], timeout=20)
        report["commands"]["tailscale_status_text"] = scrub_result(
            status_text,
            scrub_stdout=not args.include_raw_tailscale_status,
        )
        report["commands"]["tailscale_ping"] = run(
            [
                tailscale,
                "ping",
                "--c",
                str(args.ping_count),
                f"--timeout={args.ping_timeout_s}s",
                args.host,
            ],
            timeout=max(args.ping_count * args.ping_timeout_s + 5, 10),
        )
    else:
        report["findings"].append("WARN: tailscale executable was not found.")

    report["tcp"]["ssh"] = tcp_probe(args.host, args.ssh_port, args.tcp_timeout_s)

    if args.ssh_user:
        ssh = shutil.which(args.ssh_bin)
        if ssh:
            report["commands"]["ssh_probe"] = run(
                [
                    ssh,
                    "-o",
                    f"ConnectTimeout={int(args.tcp_timeout_s)}",
                    "-o",
                    "BatchMode=yes",
                    "-p",
                    str(args.ssh_port),
                    f"{args.ssh_user}@{args.host}",
                    "hostname && date",
                ],
                timeout=max(int(args.tcp_timeout_s) + 5, 10),
            )
        else:
            report["commands"]["ssh_probe"] = {"available": False}

    ping = report["commands"].get("tailscale_ping", {})
    tcp = report["tcp"]["ssh"]
    peer = report.get("peer") or {}
    if peer:
        report["findings"].append(
            "INFO: Tailscale control plane lists target "
            f"{peer.get('host_name') or peer.get('dns_name')} "
            f"online={peer.get('online')} active={peer.get('active')} "
            f"relay={peer.get('relay')} tx={peer.get('tx_bytes')} rx={peer.get('rx_bytes')}."
        )
        if peer.get("rx_bytes") == 0:
            report["findings"].append(
                "WARN: Tailscale peer rx_bytes is 0; this looks like one-way reachability."
            )
    else:
        report["findings"].append("WARN: target was not found in tailscale status JSON.")

    if ping.get("returncode") == 0:
        report["findings"].append("OK: tailscale ping reached the target.")
    else:
        report["findings"].append("WARN: tailscale ping did not reach the target.")

    if tcp.get("ok"):
        report["findings"].append(f"OK: TCP/{args.ssh_port} is reachable.")
    else:
        report["findings"].append(f"WARN: TCP/{args.ssh_port} is not reachable.")

    report["usable_for_live_work"] = (
        ping.get("returncode") == 0 and bool(tcp.get("ok"))
    )
    report["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return report


def markdown(report: dict[str, Any]) -> str:
    lines = ["# GB10 Host Access Probe", ""]
    target = report["target"]
    lines.append(
        f"Target: `{target.get('host')}`"
        + (f" / `{target.get('name')}`" if target.get("name") else "")
    )
    lines.append("")
    lines.append(f"Usable for live work: `{report.get('usable_for_live_work')}`")
    lines.append("")
    lines.append("## Findings")
    for finding in report.get("findings", []):
        lines.append(f"- {finding}")
    lines.append("")
    peer = report.get("peer")
    if peer:
        lines.append("## Tailscale Peer")
        for key in (
            "host_name",
            "dns_name",
            "online",
            "active",
            "relay",
            "cur_addr",
            "tx_bytes",
            "rx_bytes",
            "last_seen",
        ):
            lines.append(f"- `{key}`: `{peer.get(key)}`")
        lines.append("")
    lines.append("## TCP")
    ssh = report.get("tcp", {}).get("ssh", {})
    lines.append(f"- `{ssh.get('host')}:{ssh.get('port')}` ok: `{ssh.get('ok')}`")
    if ssh.get("error"):
        lines.append(f"- error: `{ssh.get('error')}`")
    lines.append("")
    lines.append("## Commands")
    for name, result in report.get("commands", {}).items():
        lines.append(f"### {name}")
        if result.get("available") is False:
            lines.append("- not available")
            continue
        lines.append(f"- returncode: `{result.get('returncode')}`")
        if result.get("error"):
            lines.append(f"- error: `{result.get('error')}`")
        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        if stdout:
            lines.extend(["", "```text", stdout, "```"])
        if stderr:
            lines.extend(["", "stderr:", "```text", stderr, "```"])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="100.113.98.11")
    parser.add_argument("--name", default="")
    parser.add_argument("--ssh-user")
    parser.add_argument("--ssh-port", type=int, default=22)
    parser.add_argument("--tcp-timeout-s", type=float, default=8.0)
    parser.add_argument("--ping-count", type=int, default=1)
    parser.add_argument("--ping-timeout-s", type=int, default=5)
    parser.add_argument("--tailscale-bin", default="tailscale")
    parser.add_argument("--ssh-bin", default="ssh")
    parser.add_argument("--output-json")
    parser.add_argument("--output-md")
    parser.add_argument("--json", action="store_true", help="print JSON instead of Markdown")
    parser.add_argument(
        "--include-raw-tailscale-status",
        action="store_true",
        help="include raw tailscale status output; off by default because it can expose tailnet metadata",
    )
    args = parser.parse_args()

    report = collect(args)
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    if args.output_md:
        Path(args.output_md).write_text(markdown(report), encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(markdown(report))
    return 0 if report.get("usable_for_live_work") else 2


if __name__ == "__main__":
    raise SystemExit(main())
