# Stereo Calibration Checklist

Use this checklist when moving beyond the Lite lane into a public Stereo
setup. A passing checklist means the public host can see and test the stereo
hardware. It does not by itself prove final training readiness.

## 1. Record Host Context

```bash
./cli/chek-ego-miner doctor --json --report-path ./artifacts/stereo/host.json
./cli/chek-ego-miner camera-probe --json --report-path ./artifacts/stereo/camera-visible.json
```

Expected result:

- host OS and architecture are present
- at least two video devices are visible for the Stereo lane
- any missing tool is explicit in the report

## 2. Prove Current-Session Camera Access

```bash
./cli/chek-ego-miner camera-probe \
  --capture-smoke \
  --json \
  --report-path ./artifacts/stereo/camera-smoke.json
```

Expected result:

- `capture_smoke.requested=true`
- `capture_smoke.ok=true`
- the report records which backend, index or name was used

If the camera is visible but smoke fails, check OS privacy permission, another
app holding the camera, USB bandwidth and the selected device index.

## 3. Generate a Charuco Target

```bash
./cli/chek-ego-miner charuco --output-dir ./artifacts/charuco
```

Print the generated board at actual size. Do not scale the PDF from the print
dialog. Keep the board flat and well lit.

## 4. Capture Calibration Frames

Collect synchronized left/right images that cover:

- center, edges and corners of both cameras
- near and far board distances
- tilt and rotation variation
- at least one stable frontal view for debugging

Save the images and calibration output under a run-specific evidence directory,
for example:

```text
artifacts/stereo/calibration-YYYYMMDD-HHMMSS/
```

## 5. Store Calibration Artifacts

A public Stereo evidence bundle should include:

- host report
- camera visibility report
- camera smoke report
- calibration images or a redacted listing of them
- stereo calibration JSON
- reprojection or validation summary when available
- notes for any skipped camera, failed frame or manual correction

## 6. Run Readiness and Public E2E Summary

```bash
./cli/chek-ego-miner readiness \
  --tier stereo \
  --capture-smoke \
  --json \
  --report-path ./artifacts/stereo/readiness.json

./cli/chek-ego-miner public-e2e \
  --tier stereo \
  --capture-smoke \
  --json \
  --report-path ./artifacts/stereo/public-e2e.json
```

The public E2E summary records host OS, tier, camera readiness, VLM policy,
local capture result and upload policy. It does not upload by default.

## What Not To Claim Yet

Do not claim public Stereo training readiness until a public bundle has the
required calibration, time-sync, SLAM benchmark and frozen threshold evidence.

Do not treat internal factory evidence as public reproducibility unless the
same public checklist and commands were used.
