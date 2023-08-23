from gym.envs.registration import register


register(
    id='reachdense-v0',
    entry_point='gym_example.envs:ReachDense',
    max_episode_steps=30,
)