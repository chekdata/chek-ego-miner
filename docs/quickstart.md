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
