"""To create an agent capable of winning at the Atari game Asteroids"""
import retro
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN

def Asteroids_main():
    env = retro.make('Asteroids-Atari2600', use_restricted_actions=retro.Actions.DISCRETE)
    model = DQN(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("Asteroids_Model")

    del model

    model = DQN.load("Asteroids_Model")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rew, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()
        env.close()

if __name__ == "__main__":
        Asteroids_main()

