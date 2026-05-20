# Public and Private Runtime Boundary

`chek-ego-miner` is the public contributor product surface. It is not a raw
dump of the internal edge runtime, and it should not become a second private
runtime with public names.

For the current cross-repo target-state table, shared contract, and multi-phone
ownership rule, see [Repo Business Contract](./repo-business-contract.md).

## Public Repo Responsibilities

This repo owns the contributor-facing path:

- public CLI names and one-step guided setup
- host checks, camera probe and readiness reports
- public Lite local E2E and bundle validation
- public Stereo and Pro checklists, diagnostics and evidence templates
- phone-vision and VLM sidecar startup paths that a contributor can run
- public docs, prompts, roadmap and validation matrix
- data safety defaults for local diagnostics

Commands in this repo should be understandable from public docs alone. Local
diagnostic commands must stay local unless the command name, docs and output
make upload explicit.

## Private Runtime Responsibilities

The private/internal edge runtime owns factory and operator engineering work
that is not generally reproducible by public contributors:

- factory-integrated device bring-up
- same-session raw/stereo/VLM engineering evidence
- production cloud, ops and deployment wiring
- deep calibration and recovery workflows
- hardware fleet observability
- commercial backend surfaces
- true GT and frozen training-readiness closure

Internal evidence can inform the public roadmap, but it cannot be overclaimed
as public reproducible evidence unless the same command, hardware expectation
and failure behavior exist here.

## Shared Capability Rule

`SLAM`, `VLM` and `time-sync` are shared capabilities. The public repo should
expose install, validation and operator feedback for them, while the private
runtime can keep deeper factory integration.

If two code paths express the same runtime behavior, they should converge into
one of these shared forms:

- a versioned runtime asset
- a shared config contract
- a public-safe template
- a generated package or bootstrap artifact
- a documented API boundary

Long-lived duplicate implementations are a bug. Public UX can differ from
factory UX, but the underlying capture-quality semantics should not drift.

## Evidence Rule

Use this wording when status depends on internal evidence:

> Internal runtime evidence shows this capability can work on controlled
> hardware. Public reproducibility remains pending until the public command,
> hardware setup, expected output and failure behavior are documented here.

Do not use internal hostnames, IPs, tokens, operator URLs or private account
state in this repository.

## Cloud Contribution Boundary

Public local checks do not upload. Cloud contribution must be a separate,
explicitly named flow with:

- account or device binding
- consent
- upload scope
- visible destination
- resumable status
- downloadable evidence or validation output

Until that path is present, public E2E means local readiness and local capture
validation, not cloud contribution.
