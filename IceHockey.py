"""To create an agent capable of winning at the Atari game IceHockey"""
import retro
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN

def IceHockey_main():
    env = retro.make('IceHockey-Atari2600', use_restricted_actions=retro.Actions.DISCRETE)
    model = DQN(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("IceHockey_Model")

    del model

    model = DQN.load("IceHockey_Model")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rew, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()
        env.close()

if __name__ == "__main__":
        IceHockey_main()

