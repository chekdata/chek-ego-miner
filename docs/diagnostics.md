# Diagnostics

## Goal

These tools help you validate your setup without needing the full private
engineering workspace.

## Scripts

### `scripts/check_host_basics.py`

Use this to print a lightweight public-safe host report for:

- OS and architecture
- Python path and version
- common tool presence
- Windows runtime helpers
- Linux or macOS video device visibility

Example:

```bash
python3 scripts/check_host_basics.py
python3 scripts/check_host_basics.py --json --report-path ./artifacts/host-basics.json
```

### `scripts/generate_charuco_a4_pdf.py`

Use this to generate a printable Charuco board for camera calibration.

Example:

```bash
python3 scripts/generate_charuco_a4_pdf.py --output-dir ./artifacts/charuco
```

Dependencies:

- `opencv-python`
- `Pillow`

### `scripts/http_load_probe.py`

Use this to send bounded HTTP traffic to a single endpoint and get a JSON report.

Example:

```bash
python3 scripts/http_load_probe.py \
  --url http://127.0.0.1:8080/health \
  --requests 20 \
  --concurrency 5
```

### `scripts/ws_stream_probe.py`

Use this to open multiple WebSocket clients and record bounded fanout metrics.

Example:

```bash
python3 scripts/ws_stream_probe.py \
  --url ws://127.0.0.1:8765/stream/fusion \
  --clients 4 \
  --duration-seconds 8
```

### `scripts/scan_public_safety.sh`

Use this before publishing changes to detect obvious internal-only patterns such
as internal IPs, hostnames, Tailscale references, or debug tokens.

Example:

```bash
./scripts/scan_public_safety.sh .
```

## Notes

- These scripts are intentionally generic and public-safe.
- They are not a replacement for the full private acceptance archive.
- `check_host_basics.py` is the first lightweight public self-check; it is not yet a full replacement for private `readiness`.
