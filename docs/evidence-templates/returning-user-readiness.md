# Returning User Readiness Evidence Template

Use this template for a contributor who previously had a working setup and
wants to confirm it still works.

## Summary

- Date:
- Hardware tier:
- Host OS:
- Repo commit:
- Prior known-good date:
- Prior known-good artifact directory:

## Commands

```bash
./cli/chek-ego-miner doctor --json --report-path ./artifacts/returning/host.json
./cli/chek-ego-miner camera-probe --capture-smoke --json --report-path ./artifacts/returning/camera-smoke.json
./cli/chek-ego-miner readiness --tier lite --json --report-path ./artifacts/returning/readiness.json
./cli/chek-ego-miner public-e2e --tier lite --json --report-path ./artifacts/returning/public-e2e.json
```

Replace `lite` with the selected tier.

## Compare With Prior Known-Good

- OS changed:
- camera device changed:
- runtime asset changed:
- VLM model changed:
- readiness blockers changed:
- public E2E changed:

## Result

- Still ready:
- Regression suspected:
- Next diagnostic:

## Notes

Do not upload just to prove readiness. Use local reports unless the contributor
is explicitly validating a documented upload flow.
