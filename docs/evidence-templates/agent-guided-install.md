# Agent-Guided Install Evidence Template

Use this template when Codex, Claude, OpenClaw or another assistant guides a
public contributor through setup.

## Summary

- Date:
- Assistant:
- Hardware tier:
- Host OS:
- Phone app installed:
- Camera hardware connected:
- Repo commit:

## Required Assistant Questions

- Which hardware tier is being installed?
- What OS is the host using?
- Is the iOS app installed?
- What camera hardware is connected now?
- Is the contributor trying local diagnostics only, or an explicit upload flow?

## Commands

```bash
./cli/chek-ego-miner doctor
./cli/chek-ego-miner camera-probe
./cli/chek-ego-miner camera-probe --capture-smoke
./cli/chek-ego-miner readiness --tier lite
./cli/chek-ego-miner public-e2e --tier lite
```

Replace `lite` with the selected tier.

## Evidence

- host report:
- camera smoke report:
- readiness report:
- public E2E report:
- final bundle validation, if local E2E was run:

## Result

- Guided setup passed:
- Contributor action still required:
- Upload attempted: no, unless a separate explicit upload flow was run

## Notes

The assistant should keep the user on public docs. If a capability requires
private credentials or factory hardware, record it as a blocker instead of
inventing a workaround.
