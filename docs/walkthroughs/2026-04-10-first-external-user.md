# Lite First-Run Walkthrough

## Scenario

- user type: first-time user
- tier: `Lite`
- example host: macOS

## Suggested steps

```bash
./cli/chek-ego-miner doctor --json
./cli/chek-ego-miner readiness --tier lite --json
./cli/chek-ego-miner quickstart --tier lite
curl -I -L -s -o /dev/null -w '%{http_code} %{url_effective}\n' 'https://testflight.apple.com/join/RrYdeDUv'
curl -I -L -s -o /dev/null -w '%{http_code} %{url_effective}\n' 'https://www.chekkk.com/humanoid/ego-dataset'
```

## What success looks like

- `doctor` prints host information and useful tier hints
- `readiness --tier lite` reports whether the host is ready
- `quickstart --tier lite` points you to the next actions
- the TestFlight link opens
- the dataset portal link opens

## If something fails

- use the hardware guide to confirm your setup matches `Lite`
- use one of the guided prompts to continue one step at a time
- fix readiness issues before moving to advanced setup
