# Quickstart

## 1. Install the iOS app

- [TestFlight](https://testflight.apple.com/join/RrYdeDUv)

## 2. Pick your tier

- `Lite`
- `Stereo`
- `Pro`

See [Hardware Guide](./hardware.md).

## 3. Run the public CLI doctor

```bash
./cli/chek-ego-miner doctor
```

## 4. Run readiness for your tier

```bash
./cli/chek-ego-miner readiness --tier lite
./cli/chek-ego-miner readiness --tier stereo
./cli/chek-ego-miner readiness --tier pro
```

## 5. Ask an agent to guide you

Pick a prompt from:

- `prompts/install-lite.md`
- `prompts/install-stereo.md`
- `prompts/install-pro-edge.md`

## 6. If you need calibration

```bash
./cli/chek-ego-miner charuco --output-dir ./artifacts/charuco
```

## 7. Reinstall a Linux or macOS basic host from the public repo

```bash
./cli/chek-ego-miner install \
  --profile basic \
  --apply \
  --system-install \
  --enable-services
```

## 8. Enable the local phone-vision sidecar

```bash
python3 -m pip install --user --break-system-packages -r scripts/edge_phone_vision_requirements.txt
./cli/chek-ego-miner fetch-phone-vision-models --json
./scripts/start_edge_phone_vision_service.sh
```

## 9. Run the verified basic end-to-end flow

```bash
./cli/chek-ego-miner basic-e2e \
  --edge-base-url http://127.0.0.1:8080 \
  --edge-token chek-ego-miner-local-token \
  --trip-id trip-public-basic-e2e \
  --session-id sess-public-basic-e2e \
  --output-dir ./artifacts/basic-e2e \
  --json
```

Expected result:

- `ok: true`
- `validation.ok: true`
- `validation.score_percent: 100.0`
- `public_download/demo_capture_bundle.json` exists

Notes:

- This exact lane has been verified on a dedicated `Linux x86_64` edge host.
- This exact lane has also been verified on a local `macOS arm64` developer machine.
- On macOS, `install --system-install --enable-services` now auto-stages the runtime under `~/.chek-edge/runtime/macos/basic` so `launchd` does not depend on Desktop/Documents permissions.
- The phone-vision start script auto-selects a compatible interpreter when `python3` itself is not usable.
- `time_sync_samples` may remain empty on the single-phone basic lane; it is currently reported as an advisory rather than a blocker.

## 10. Deliver the public Pro Jetson path

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

If your Jetson already has a working GPU VLM venv and local SmolVLM model
cache, or if you only want to wire the VLM portion instead of the full
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

- `jetson-professional-bootstrap` wires stereo calibration, the Wi-Fi sensing model and binary, runtime binaries, workstation dist, and the existing Jetson VLM runtime into the public repo layout.
- `fetch-vlm-models` downloads the core Hugging Face files only, not the extra ONNX variants.
- `vlm-start` auto-selects a compatible Python interpreter and looks for `SmolVLM2-500M` plus `SmolVLM2-256M` under `model-candidates/huggingface/`.
- The repo now ships `edge_vlm_sidecar.py` directly as a public runtime asset.
- The Jetson bootstrap path now also has a live acceptance pass: `readiness --tier pro`, `jetson-professional-bootstrap`, public-path `systemd` / `systemd-user` services entering `active`, live `/health`, live `/association/hint`, live `/api/v1/stream/status`, and a real `/infer` response on the Jetson host.
