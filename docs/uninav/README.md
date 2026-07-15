# UniNav

UniNav is a low-compute reinforcement learning navigation project built on top of UniLab.

## Project Goal
Train a lightweight reinforcement learning policy for differential-drive robot navigation using UniLab.

## Version0: Navigation MVP
The first version contains:
- MuJoCo simulation backend
- Differential-frive robot
- Random start position
- Random goal position
- PointGoal navigation task
- PPO training
- Linear velocity and angular velocity actions
- success-rate evaluation

## Obserbation
The first versinos uses:
- Relative goal distance
- Relative goal direction
- Current linear velocity
- Current angular velocity

LiDAR will be introduced in a later milestone.

## Action
The policy outputs:
- Linear velocity `v`
- Angular velocity `w`

## Success Condition
The robot succeeds when its distance to the target is below a fixed threshld

## Roadmap
1. Run thr original UniLab PPO example
2. Understand the UniLab environment contract
3. Build a differential-drive robot
4. Implement an obstacle-free PointGoal task
5. Train the first PPO poicy
6. Add obstacles and LiDAR
7. Add evaluation metrics
8. Deploy the trained policy to ROS2

