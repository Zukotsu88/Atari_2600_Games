"""To create an agent capable of winning at the Atari game Riverraid"""
import retro
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN

def Riverraid_main():
    env = retro.make('Riverraid-Atari2600', use_restricted_actions=retro.Actions.DISCRETE)
    model = DQN(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("Riverraid_Model")

    del model

    model = DQN.load("Riverraid_Model")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rew, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()
        env.close()

if __name__ == "__main__":
        Riverraid_main()

