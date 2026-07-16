# Install this bundle

Extract the archive into:

```text
/home/xiaodudu/robot_study/UniLab-Navigation
```

Then run:

```bash
cd /home/xiaodudu/robot_study/UniLab-Navigation
bash scripts/install_codex_skills.sh
```

Restart Codex and begin with:

```text
Use $unilab-navigation-orchestrator. Read AGENTS.md, inspect the current branch,
and implement M5.2 fixed-episode evaluation completely. Continue through tests,
a real MuJoCo smoke evaluation, one milestone commit, and push.
```

The tracked skill sources remain under `codex/skills/`. The installer copies them
to `${CODEX_HOME:-$HOME/.codex}/skills`, which is where Codex discovers user skills.
