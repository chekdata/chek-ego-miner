# Camera Troubleshooting Evidence Template

Use this template when a camera is listed by the OS but capture does not work,
or when the expected stereo device count is not visible.

## Summary

- Date:
- Hardware tier:
- Host OS:
- Camera model:
- Connection type:
- Repo commit:

## Commands

```bash
./cli/chek-ego-miner camera-probe --json --report-path ./artifacts/camera/visible.json
./cli/chek-ego-miner camera-probe --capture-smoke --json --report-path ./artifacts/camera/smoke.json
./cli/chek-ego-miner readiness --tier stereo --capture-smoke --json --report-path ./artifacts/camera/readiness.json
```

Use the actual target tier.

## Checks

- OS privacy permission granted:
- another app was using camera:
- correct device index selected:
- USB cable or hub changed:
- powered hub tried:
- lower resolution or frame rate tried:
- camera firmware or vendor tool checked:

## Result

- OS-visible devices:
- capture smoke result:
- readiness blockers:
- suspected root cause:
- next action:

## Notes

Camera visibility is weaker evidence than frame capture. For Stereo and Pro,
record a current-session frame smoke result before claiming readiness.
