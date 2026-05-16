# Interrupted Install Resume Evidence Template

Use this template when validating that a contributor can safely resume after an
interrupted install or setup session.

## Summary

- Date:
- Hardware tier:
- Host OS:
- Repo commit:
- Last successful step before interruption:
- Interruption type:

## Resume Commands

```bash
./cli/chek-ego-miner doctor --json --report-path ./artifacts/resume/host.json
./cli/chek-ego-miner readiness --tier lite --json --report-path ./artifacts/resume/readiness-before.json
./cli/chek-ego-miner public-e2e --tier lite --json --report-path ./artifacts/resume/public-e2e.json
```

Replace `lite` with the selected tier. Add `--capture-smoke` when validating
camera access after the interruption.

## Evidence To Capture

- whether existing runtime files were detected
- whether reinstall was avoided or intentionally repeated
- readiness blockers before and after resume
- final public E2E report

## Result

- Resume passed:
- Manual cleanup required:
- Commands that were safe to rerun:
- Commands that should not be rerun automatically:

## Notes

The assistant or operator should explain each command before running it and
should not delete existing runtime state unless the contributor confirms it.
