# GB10 Host Access Probe

Target: `100.113.98.11`

Usable for live work: `False`

## Findings
- INFO: Tailscale control plane lists target thinkstationpgx-00b4 online=True active=True relay=tok tx=281112 rx=0.
- WARN: Tailscale peer rx_bytes is 0; this looks like one-way reachability.
- WARN: tailscale ping did not reach the target.
- WARN: TCP/22 is not reachable.
- WARN: SSH probe did not succeed.

## Tailscale Peer
- `host_name`: `thinkstationpgx-00b4`
- `dns_name`: `thinkstationpgx-00b4.tail740c8d.ts.net.`
- `online`: `True`
- `active`: `True`
- `relay`: `tok`
- `cur_addr`: ``
- `tx_bytes`: `281112`
- `rx_bytes`: `0`
- `last_seen`: `2026-06-08T20:20:47.1Z`

## TCP
- `100.113.98.11:22` ok: `False`
- error: `TimeoutError('timed out')`

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
- returncode: `1`

```text
ping "100.113.98.11" timed out
```

stderr:
```text
no reply
```
### ssh_probe
- returncode: `255`

stderr:
```text
ssh: connect to host 100.113.98.11 port 22: Connection timed out
```
