# 2026-04-10 First External-User Walkthrough

## Scenario

- User type: first-time public user
- Tier: `Lite`
- Host used for walkthrough: local macOS host

## Steps executed

```bash
cd /Users/jasonhong/Desktop/开发项目/chek-ego-miner

./cli/chek-ego-miner doctor --json
./cli/chek-ego-miner readiness --tier lite --json
./cli/chek-ego-miner quickstart --tier lite
./scripts/scan_public_safety.sh .
curl -I -L -s -o /dev/null -w '%{http_code} %{url_effective}\n' 'https://testflight.apple.com/join/RrYdeDUv'
curl -I -L -s -o /dev/null -w '%{http_code} %{url_effective}\n' 'https://www-dev.chekkk.com/humanoid/ego-dataset'
```

## Result

- public CLI `doctor` returned a valid host report
  - `system=Darwin`
  - `machine=arm64`
  - `tier_hints=["lite_possible"]`
- public CLI `readiness --tier lite` returned `ready=true` on the current host
- quickstart lane printed the next actions clearly
- public safety scan passed:
  - `scan_public_safety: no internal-host/token patterns found`
- both primary external links resolved successfully during this walkthrough:
  - TestFlight: `200`
  - dataset portal: `200`
- public Charuco generation lane also passed during the same validation batch:
  - `/private/tmp/chek-ego-miner-charuco-test/charuco_a4_portrait_8x6_24mm_18mm_dict_4x4_50.pdf`
  - `/private/tmp/chek-ego-miner-charuco-test/charuco_a4_portrait_8x6_24mm_18mm_dict_4x4_50.png`

## Notes

- This walkthrough validates the first public “arrive, read, self-check, and continue” lane.
- It does not claim that every hardware tier is fully automated yet.
