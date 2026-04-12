# Agent-Guided Lite Walkthrough

## Scenario

- tier: `Lite`
- guidance style: one step at a time
- prompt source: `prompts/install-lite.md`

## Recommended flow

1. Confirm the tier and host OS.
2. Run `doctor`.
3. Run `readiness --tier lite`.
4. Open the app download link and hardware guide.
5. Continue step by step instead of jumping to advanced setup.

## Commands you will usually see

```bash
./cli/chek-ego-miner doctor
./cli/chek-ego-miner readiness --tier lite
./cli/chek-ego-miner quickstart --tier lite
```

## What good guidance looks like

- the assistant asks one question at a time
- the assistant explains what each command is checking
- the assistant waits for your result before continuing
- the assistant keeps you on the Lite path until you are ready for more
