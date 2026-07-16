# Benchmark Result Schema

Every benchmark JSON should include:

```json
{
  "schema_version": 1,
  "git_sha": "string",
  "dirty": false,
  "timestamp_utc": "ISO-8601",
  "hardware": {
    "cpu": "string",
    "gpu": "string",
    "ram_bytes": 0
  },
  "software": {
    "python": "string",
    "torch": "string",
    "mujoco": "string",
    "unilab": "string"
  },
  "experiment": {
    "task": "DiffDrivePointGoal",
    "backend": "mujoco",
    "policy": "zero|random|heuristic|ppo|sac|td3",
    "seed": 1,
    "num_envs": 4096,
    "num_episodes": 4096,
    "checkpoint": null
  },
  "task_metrics": {
    "success_rate": 0.0,
    "timeout_rate": 0.0,
    "collision_rate": null,
    "mean_final_distance": 0.0,
    "mean_episode_length": 0.0
  },
  "performance": {
    "steps_per_second": 0.0,
    "collection_time_sec": 0.0,
    "update_time_sec": null,
    "wall_time_sec": 0.0,
    "peak_gpu_memory_bytes": null
  }
}
```

Do not replace raw per-episode output with only averages. Save either the per-episode records or a referenced compressed file.
