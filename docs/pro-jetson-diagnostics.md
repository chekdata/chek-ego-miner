# Pro Jetson Diagnostics

Use this guide for public Pro lane bring-up on a Jetson-class Linux edge host.
It is a diagnostic and evidence path, not a promise that every Pro training
gate is already frozen.

## 1. Host and Camera Checks

```bash
./cli/chek-ego-miner doctor --json --report-path ./artifacts/pro/host.json
./cli/chek-ego-miner camera-probe --json --report-path ./artifacts/pro/camera-visible.json
./cli/chek-ego-miner camera-probe \
  --capture-smoke \
  --json \
  --report-path ./artifacts/pro/camera-smoke.json
```

Expected result:

- Linux host facts are present
- camera devices are visible
- at least one selected camera can read a frame in the current session

## 2. Bootstrap Runtime Assets

For the full professional path:

```bash
./cli/chek-ego-miner jetson-professional-bootstrap -- --force
./cli/chek-ego-miner install \
  --profile professional \
  --apply \
  --system-install \
  --runtime-edge-root "$PWD"
```

For a VLM-only Pro bring-up on a host that already has the needed GPU model
environment:

```bash
./cli/chek-ego-miner jetson-vlm-bootstrap -- --force
./cli/chek-ego-miner service-install \
  --profile professional \
  --service chek-edge-vlm-sidecar \
  --enable \
  --runtime-edge-root "$PWD"
```

## 3. Fetch or Wire VLM Models

```bash
python3 -m pip install --user -r scripts/edge_vlm_requirements.txt
./cli/chek-ego-miner fetch-vlm-models --json --report-path ./artifacts/pro/vlm-models.json
```

`fetch-vlm-models` downloads the Hugging Face files required by the public VLM
sidecar. If the host already has an approved local model cache, record the
cache path in the evidence notes instead of copying secrets or private paths
into the repo.

## 4. Start and Probe Services

```bash
./cli/chek-ego-miner vlm-start
```

In another shell, record health and inference evidence from the configured
local endpoint:

```bash
curl -fsS http://127.0.0.1:8091/health
curl -fsS http://127.0.0.1:8091/infer \
  -H 'content-type: application/json' \
  --data '{"prompt":"Describe the scene for a readiness check.","image_b64":""}'
```

Use the actual local port configured by the sidecar. Do not paste private
tokens or internal URLs into public evidence.

## 5. Run Strict Readiness

```bash
./cli/chek-ego-miner readiness \
  --tier pro \
  --capture-smoke \
  --json \
  --report-path ./artifacts/pro/readiness.json

./cli/chek-ego-miner public-e2e \
  --tier pro \
  --capture-smoke \
  --json \
  --report-path ./artifacts/pro/public-e2e.json
```

For Pro, the public E2E report marks VLM as required for the tier. The report
also makes upload status explicit; it remains disabled unless the operator runs
a separate documented bind/upload flow.

## Evidence Checklist

Keep a run directory with:

- host report
- camera visibility and smoke reports
- bootstrap command log
- model fetch or model-cache evidence
- VLM health and inference response
- readiness report
- public E2E report
- any stereo calibration or Wi-Fi sensing evidence available for this host

## What Not To Claim Yet

Do not claim final Pro training readiness until the session has true GT or
approved manual GT, multi-round baselines, frozen thresholds and passing
training gates.

Do not claim that private same-session evidence is public reproducibility unless
the same result is reproduced with public commands and non-secret evidence.
