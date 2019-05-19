import gym

env = gym.make('CartPole-v0')
for _ in range(1000):
    env.reset()
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()