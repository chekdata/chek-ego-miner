# Quickstart

## 1. Install the iOS app

- [TestFlight](https://testflight.apple.com/join/RrYdeDUv)

## 2. Choose your tier

- `Lite`
- `Stereo`
- `Pro`

See [Hardware Guide](./hardware.md).

## 3. Run the CLI doctor

```bash
./cli/chek-ego-miner doctor
```

## 4. Run readiness for your tier

```bash
./cli/chek-ego-miner readiness --tier lite
./cli/chek-ego-miner readiness --tier stereo
./cli/chek-ego-miner readiness --tier pro
```

## 5. Probe local cameras

```bash
./cli/chek-ego-miner camera-probe
./cli/chek-ego-miner camera-probe --capture-smoke
```

The first command checks OS-visible camera devices. The second command also
tries to open one camera and read a frame from the current terminal session.

## 6. Get guided help

Pick a prompt from:

- `prompts/install-lite.md`
- `prompts/install-stereo.md`
- `prompts/install-pro-edge.md`

## 7. If you need calibration

```bash
./cli/chek-ego-miner charuco --output-dir ./artifacts/charuco
```

## 8. Set up Lite on Linux or macOS

```bash
./cli/chek-ego-miner install \
  --profile basic \
  --apply \
  --system-install \
  --enable-services
```

## 9. Enable the local phone-vision sidecar

```bash
python3 -m pip install --user --break-system-packages -r scripts/edge_phone_vision_requirements.txt
./cli/chek-ego-miner fetch-phone-vision-models --json
./scripts/start_edge_phone_vision_service.sh
```

## 10. Run the basic capture flow

```bash
./cli/chek-ego-miner basic-e2e \
  --edge-base-url http://127.0.0.1:8080 \
  --edge-token chek-ego-miner-local-token \
  --trip-id trip-public-basic-e2e \
  --session-id sess-public-basic-e2e \
  --output-dir ./artifacts/basic-e2e \
  --json
```

You should see:

- `ok: true`
- `validation.ok: true`
- `validation.score_percent: 100.0`
- `public_download/demo_capture_bundle.json` exists

Notes:

- This path is intended for `Linux x86_64` and `macOS arm64` basic hosts.
- On macOS, `install --system-install --enable-services` auto-stages the
  runtime under `~/.chek-edge/runtime/macos/basic`.
- The phone-vision start script auto-selects a compatible interpreter when
  `python3` itself is not usable.
- `time_sync_samples` may remain empty on the single-phone basic path.

## 11. Set up Pro on Jetson

```bash
./cli/chek-ego-miner jetson-professional-bootstrap -- --force
./cli/chek-ego-miner install \
  --profile professional \
  --apply \
  --system-install \
  --runtime-edge-root "$PWD"

python3 -m pip install --user -r scripts/edge_vlm_requirements.txt
./cli/chek-ego-miner fetch-vlm-models --json
./cli/chek-ego-miner vlm-start
```

If your Jetson already has a working GPU VLM environment and local SmolVLM
model cache, or if you only want to wire the VLM portion instead of the full
professional asset set, use:

```bash
./cli/chek-ego-miner jetson-vlm-bootstrap -- --force
./cli/chek-ego-miner service-install \
  --profile professional \
  --service chek-edge-vlm-sidecar \
  --enable \
  --runtime-edge-root "$PWD"
```

Notes:

- `jetson-professional-bootstrap` connects stereo calibration, the Wi-Fi
  sensing model and binary, runtime binaries, workstation dist, and an
  existing Jetson VLM environment.
- `fetch-vlm-models` downloads the core Hugging Face files only.
- `vlm-start` auto-selects a compatible Python interpreter and looks for
  `SmolVLM2-500M` plus `SmolVLM2-256M` under `model-candidates/huggingface/`.
- A successful Jetson bring-up should end with required services in `active`
  and working `/health`, `/association/hint`, `/api/v1/stream/status`, and
  `/infer` responses on the host.
