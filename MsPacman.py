"""To create an agent capable of winning at the Atari game MsPacman"""
import retro
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN

def MsPacman_main():
    env = retro.make('MsPacman-Atari2600', use_restricted_actions=retro.Actions.DISCRETE)
    model = DQN(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("MsPacman_Model")

    del model

    model = DQN.load("MsPacman_Model")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rew, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()
        env.close()

if __name__ == "__main__":
        MsPacman_main()

