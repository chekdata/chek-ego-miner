# First Contributor Evidence Template

Use this template for a clean first setup on a contributor machine.

## Summary

- Date:
- Contributor alias:
- Hardware tier: `Lite` / `Stereo` / `Pro`
- Host OS and version:
- CPU architecture:
- Phone model:
- Camera hardware:
- Repo commit:

## Commands Run

```bash
./cli/chek-ego-miner doctor --json --report-path ./artifacts/first/host.json
./cli/chek-ego-miner camera-probe --capture-smoke --json --report-path ./artifacts/first/camera-smoke.json
./cli/chek-ego-miner readiness --tier lite --json --report-path ./artifacts/first/readiness.json
./cli/chek-ego-miner public-e2e --tier lite --json --report-path ./artifacts/first/public-e2e.json
```

Replace `lite` with the selected tier. Use `--capture-smoke` for `Stereo` and
`Pro` evidence.

## Expected Artifacts

- `host.json`
- `camera-smoke.json`
- `readiness.json`
- `public-e2e.json`
- basic local bundle evidence if `--run-basic-e2e` was used

## Result

- Passed:
- Blockers:
- Warnings:
- Upload attempted: no, unless a separate explicit upload flow was run

## Notes

Do not include internal hostnames, private IPs, tokens, account cookies or
operator-only URLs.
