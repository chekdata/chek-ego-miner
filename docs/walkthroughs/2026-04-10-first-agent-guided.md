# 2026-04-10 First Agent-Guided Walkthrough

## Scenario

- Agent style: Codex-style step-by-step guidance
- Tier: `Lite`
- Prompt source: `prompts/install-lite.md`

## Flow validated

1. Determine tier and host OS.
2. Run host doctor.
3. Run tier readiness.
4. Point the user to the app download and hardware guide.
5. Continue step by step instead of skipping to advanced setup.

## Commands executed

```bash
cd /Users/jasonhong/Desktop/开发项目/chek-ego-miner

./cli/chek-ego-miner doctor
./cli/chek-ego-miner readiness --tier lite
./cli/chek-ego-miner quickstart --tier lite
```

## Result

- the public prompt and docs supported a real step-by-step agent workflow
- the public CLI gave the agent enough host context to continue safely
- the first recommended lane is now “app download -> doctor -> readiness -> quickstart -> agent guidance”
- validated commands all passed on the current macOS host:
  - `./cli/chek-ego-miner doctor`
  - `./cli/chek-ego-miner readiness --tier lite`
  - `./cli/chek-ego-miner quickstart --tier lite`
