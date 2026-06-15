[简体中文](./README.zh-CN.md)

# CHEK EGO Miner / Qingkong Miker

Use an iPhone and a computer to capture first-person everyday actions for embodied-AI data collection.

CHEK EGO Miner is primarily for public collectors, not only robotics engineers. A collector can record real work or daily tasks from the “I am doing this” point of view, then use CHEK / Qingkong Miker to follow the task flow for capture, upload, review, and task-based rewards when a task supports them.

This repository is the public documentation hub. It contains download, setup, capture, hardware, privacy, and troubleshooting docs. It does not publish source code, build scripts, internal runtime code, old product logic, or temporary test materials.

## One-Sentence Idea

The internet era turned text, photos, and videos into AI training material. Embodied AI now needs human actions, viewpoints, decisions, and work sequences so robots can learn how real people handle the physical world.

## Who This Repo Is For

1. **Public collectors** who want to use a phone and a computer to participate in EGO data tasks.
2. **Community readers and media explainers** who want to understand why EGO data collection is being described as a new kind of data mining.
3. **Device and scene partners** who need public hardware, privacy, and delivery boundaries.
4. **Documentation contributors** who want to improve public-safe setup guides, screenshots, hardware notes, and troubleshooting.

Developers should note that this is not a source-code repository. Do not add runtime code, internal service scripts, private deployment instructions, credentials, or private logs here.

## Start Here

| Goal | Link |
| --- | --- |
| Download the clients | [Download guide](./docs/download.md) |
| Start your first phone-and-computer collection | [Quick start](./docs/quickstart.md) |
| Prepare phone, computer, mount, camera, or IMU hardware | [Hardware guide](./docs/hardware.md) |
| Record a real EGO data session | [Capture guide](./docs/capture-guide.md) |
| Understand what a session will save | [Delivery contract](./docs/delivery-contract.md) |
| Fix download, preview, storage, or device-recognition issues | [Troubleshooting](./docs/troubleshooting.md) |
| Understand consent and public-screenshot boundaries | [Privacy](./docs/privacy.md) |
| Read common questions | [FAQ](./docs/faq.md) |

## Official Downloads

| Platform | Official entry | Notes |
| --- | --- | --- |
| Desktop `macOS / Windows / Linux` | [smart-download](https://www.chekkk.com/smart-download) | Open on a desktop browser to reach the desktop-client branch. |
| `iOS` | [TestFlight](https://testflight.apple.com/join/RrYdeDUv) | Distributed through TestFlight for now. |
| `Android` | [smart-download](https://www.chekkk.com/smart-download) | Routes through app-market flows first and falls back to APK download. |

## Why This Is Not Just Video Recording

A normal video ends when it is shot. EGO data collection tries to turn a real human task into a governed, reviewable, reusable data asset. That means the workflow also cares about:

- which task and scene the collector is working on;
- whether the phone or camera recorded a stable first-person view;
- whether the session has a clear capture, upload, and review flow;
- whether privacy, consent, quality, and delivery checks are satisfied;
- whether the resulting data can be searched, reused, and rewarded according to task rules.

## About Rewards

Public explainers sometimes describe this as data mining because real human actions and experience can become robot-training material. Actual rewards, review rules, and settlement terms depend on the specific task and platform policy. This repository does not promise a fixed hourly income.

## Repository Boundary

The repository name stays `chek-ego-miner`, while the public user-facing name is “CHEK EGO Miner / Qingkong Miker”.

This repository keeps only public documentation:

- download and install instructions;
- phone, computer, mount, camera, and IMU guidance;
- EGO capture workflow;
- privacy, consent, safety, and public issue boundaries;
- troubleshooting for ordinary users.

It does not keep:

- source code;
- internal runtime code;
- private deployment scripts;
- old product or temporary validation logic;
- secrets, accounts, private URLs, or non-public logs.

## License Boundary

Documentation content is open for use. App binaries, services, hardware protocols, trademarks, and official release materials stay outside that open boundary.
