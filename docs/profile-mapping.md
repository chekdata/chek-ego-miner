# Hardware Tiers and Internal Profile Mapping

## Goal

Explain the public hardware tiers in plain language while still preserving the
mapping to the internal profile vocabulary used in private engineering work.

## Mapping

| Public tier | Typical hardware | Internal profile mapping | Current public expectation |
| --- | --- | --- | --- |
| `Lite` | computer + your own camera + iPhone | closest to `basic` | easiest way to get started |
| `Stereo` | computer + stereo camera + iPhone | closest to `enhanced` | better depth and spatial quality |
| `Pro` | edge machine + stereo camera + iPhone | closest to `professional` | dedicated capture and higher-throughput workflows |

## Why public docs use tiers first

The public repo is optimized for operators and contributors, not for internal
engineering vocabulary. Most people can choose hardware much faster than they
can reason about profile manifests.

## Notes

- The internal profile system remains useful for deeper runtime packaging.
- Public docs should only expose the mapping where it helps understanding.
