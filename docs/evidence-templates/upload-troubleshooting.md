# Upload Troubleshooting Evidence Template

Use this template only for an explicit upload or cloud contribution flow. Local
readiness, camera probe and public E2E do not upload by default.

## Summary

- Date:
- Hardware tier:
- Host OS:
- Repo commit:
- Account or device binding confirmed:
- Consent confirmed:
- Upload scope:
- Destination:

## Local Preflight

```bash
./cli/chek-ego-miner public-e2e --tier lite --json --report-path ./artifacts/upload/preflight.json
```

Replace `lite` with the selected tier. Add `--capture-smoke` for Stereo or Pro.

## Upload Flow Evidence

- command or UI path used:
- upload request id:
- resumable status:
- server response status:
- worker or processing status:
- downloadable bundle or portal item:
- validation report:

## Result

- Upload succeeded:
- Download succeeded:
- Validation succeeded:
- Blocker:

## Notes

Do not paste bearer tokens, cookies, payment data, private operator URLs or
internal hostnames into this file. Redact account identifiers unless the
contributor has approved sharing them.
