---
name: unilab-navigation-orchestrator
description: Orchestrate milestone-by-milestone implementation of the UniLab Navigation and SLAM project. Use for broad requests to continue the project, decide the next milestone, coordinate navigation RL, SLAM adapters, evaluation, benchmarking, testing, and Git delivery in the UniLab-Navigation repository.
metadata:
  short-description: Orchestrate the UniLab navigation project
---

# UniLab Navigation Orchestrator

## Start

1. Locate the repository root. Prefer `/home/xiaodudu/robot_study/UniLab-Navigation`.
2. Read `AGENTS.md`.
3. Read `docs/codex/implementation-roadmap.md`.
4. Run `scripts/check_codex_project_state.sh`.
5. Inspect the files and tests directly related to the next milestone.

Do not assume a milestone is incomplete merely because it appears in the roadmap. Verify the code and Git history.

## Select work

Choose the earliest incomplete milestone whose prerequisites pass.

Current expected next milestone: M5.2 fixed-episode evaluator.

For each milestone, write a compact implementation contract before editing:

- objective;
- files to add/change;
- public interfaces;
- invariants;
- tests;
- smoke command;
- acceptance criteria;
- commit scope.

Then implement the entire milestone. Do not stop after only creating one helper unless a real error blocks further progress.

## Delegate by domain

Read and apply:

- `../unilab-navigation-task/SKILL.md` for environment/task/backend changes;
- `../unilab-rl-adapter/SKILL.md` for algorithm integration;
- `../unilab-slam-adapter/SKILL.md` for estimator and sensor integration;
- `../unilab-benchmark/SKILL.md` for evaluation and performance claims;
- `../unilab-test-release/SKILL.md` before commit and push.

## Evidence rules

A feature is complete only when:

1. implementation exists;
2. targeted tests pass;
3. full navigation tests pass;
4. a relevant runtime smoke test passes;
5. generated artifacts are excluded from Git;
6. the result is committed with one milestone-sized commit.

A learning claim additionally needs independent fixed-episode evaluation.

A performance claim additionally needs the controlled benchmark protocol.

## Progress updates

During long work, report:

- the interface being implemented;
- the first confirmed defect or missing dependency;
- the latest passing test boundary;
- any deviation from the milestone contract.

Do not narrate low-level commands one by one.
