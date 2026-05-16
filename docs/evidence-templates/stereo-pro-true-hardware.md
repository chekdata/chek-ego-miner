# Stereo and Pro True-Hardware Evidence Template

Use this template for public evidence that a real Stereo or Pro hardware setup
was tested, rather than inferred from internal runtime evidence.

## Summary

- Date:
- Lane: `Stereo` / `Pro`
- Host OS:
- Host architecture:
- Edge hardware model:
- Camera model:
- Phone model:
- Repo commit:

## Required Public Commands

```bash
./cli/chek-ego-miner doctor --json --report-path ./artifacts/true-hardware/host.json
./cli/chek-ego-miner camera-probe --capture-smoke --json --report-path ./artifacts/true-hardware/camera-smoke.json
./cli/chek-ego-miner readiness --tier stereo --capture-smoke --json --report-path ./artifacts/true-hardware/readiness.json
./cli/chek-ego-miner public-e2e --tier stereo --capture-smoke --json --report-path ./artifacts/true-hardware/public-e2e.json
```

Use `--tier pro` for Pro evidence.

## Additional Pro Evidence

- VLM model fetch or model-cache evidence:
- VLM health response:
- VLM inference response:
- stereo producer status:
- Wi-Fi sensing status, if used:
- service status:

## Result

- Public commands passed:
- Current-session camera frame read:
- Calibration available:
- Local bundle validated:
- Upload attempted: no, unless a separate explicit upload flow was run

## Notes

This template is the evidence needed before closing public hardware issues for
Stereo or Pro. Internal factory evidence can be cited as background, but it
does not replace public command output.
