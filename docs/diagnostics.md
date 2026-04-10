# Diagnostics

## Goal

These tools help you validate your setup from the public repo itself.

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

### `scripts/fetch_phone_vision_models.py`

Use this to download the public model files required by the local phone-vision
sidecar.

Example:

```bash
./cli/chek-ego-miner fetch-phone-vision-models --json
```

### `scripts/start_edge_phone_vision_service.sh`

Use this to start the local phone-vision sidecar that turns iPhone ingress
frames into `capture_pose` artifacts on a host machine.

Example:

```bash
python3 -m pip install --user --break-system-packages -r scripts/edge_phone_vision_requirements.txt
./scripts/start_edge_phone_vision_service.sh
```

Notes:

- The start script auto-selects a compatible interpreter from the configured
  venv, Isaac Python, or local `python3.10` / `python3.11` / `python3.12` /
  `python3.13` / `python3`.
- On Homebrew-managed macOS Python, `pip install --user` may require
  `--break-system-packages`, or you can install the dependencies into a
  compatible interpreter such as `python3.10`.

### `scripts/fetch_vlm_models.py`

Use this to download the public VLM model files needed by the Jetson `Pro`
sidecar. The fetcher only downloads the core Hugging Face files required by
`transformers`, not the extra ONNX variants.

Example:

```bash
./cli/chek-ego-miner fetch-vlm-models --json
./cli/chek-ego-miner fetch-vlm-models \
  --models-root /tmp/chek-vlm-models \
  --primary-model-id SmolVLM2-256M \
  --skip-fallback \
  --json
```

### `scripts/start_edge_vlm_sidecar.sh`

Use this to start the public VLM sidecar that powers `professional` edge-host
semantic indexing.

Example:

```bash
python3 -m pip install --user -r scripts/edge_vlm_requirements.txt
./cli/chek-ego-miner fetch-vlm-models --json
./cli/chek-ego-miner vlm-start
```

Notes:

- The start script auto-selects a compatible interpreter and checks for
  `torch`, `transformers`, `huggingface_hub`, `Pillow`, and `num2words`.
- The default model locations are under `model-candidates/huggingface/`.
- A local smoke was re-verified with `SmolVLM2-256M` returning a real caption
  from `/infer`.

### `scripts/run_basic_e2e.py`

Use this to run the verified synthetic `basic` flow:

- send synthetic phone ingress packets
- download the edge bundle from `/live-preview/file/...`
- export a public-safe download subset
- validate the resulting bundle contract

Example:

```bash
./cli/chek-ego-miner basic-e2e \
  --edge-base-url http://127.0.0.1:8080 \
  --edge-token chek-ego-miner-local-token \
  --trip-id trip-public-basic-e2e \
  --session-id sess-public-basic-e2e \
  --output-dir ./artifacts/basic-e2e \
  --json
```

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
- `check_host_basics.py` is the first lightweight public self-check.
- The public repo now also exposes a real `basic` synthetic end-to-end lane for dedicated Linux edge hosts.
- The public repo now also ships a real `professional` VLM model-fetch and sidecar path.
