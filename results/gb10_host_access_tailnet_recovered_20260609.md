# GB10 Host Access Probe

Target: `100.113.98.11`

Usable for live work: `True`

## Findings
- INFO: Tailscale control plane lists target thinkstationpgx-00b4 online=True active=False relay=tok tx=28000 rx=61960.
- OK: tailscale ping reached the target.
- OK: TCP/22 is reachable.
- OK: SSH probe succeeded.

## Tailscale Peer
- `host_name`: `thinkstationpgx-00b4`
- `dns_name`: `thinkstationpgx-00b4.tail740c8d.ts.net.`
- `online`: `True`
- `active`: `False`
- `relay`: `tok`
- `cur_addr`: ``
- `tx_bytes`: `28000`
- `rx_bytes`: `61960`
- `last_seen`: `0001-01-01T00:00:00Z`

## TCP
- `100.113.98.11:22` ok: `True`

## Commands
### tailscale_status_json
- returncode: `0`

```text
<redacted; target peer summary recorded separately>
```
### tailscale_status_text
- returncode: `0`

```text
<redacted; target peer summary recorded separately>
```
### tailscale_ping
- returncode: `0`

```text
pong from thinkstationpgx-00b4 (100.113.98.11) via [240d:f:d4c:3d00:f506:e566:79b7:abd4]:41641 in 2ms
```
### ssh_probe
- returncode: `0`

```text
thinkstationpgx-00b4
Tue Jun  9 02:32:40 PM JST 2026
```

## Additional Root Probe

Separate key-based SSH probe:

```powershell
ssh -o BatchMode=yes -o ConnectTimeout=8 root@thinkstationpgx-00b4 "hostname && whoami && uname -a"
```

Result: returned `thinkstationpgx-00b4`, user `root`, and Linux/aarch64 kernel details.
