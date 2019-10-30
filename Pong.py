"""To create an agent capable of winning at the Atari game Pong"""
import retro
from stable_baselines.deepq.policies import CnnPolicy
from stable_baselines import DQN

def Pong_main():
    env = retro.make('Pong-Atari2600', use_restricted_actions=retro.Actions.DISCRETE)
    model = DQN(CnnPolicy, env, verbose=1)
    model.learn(total_timesteps=25000)
    model.save("Pong_Model")

    del model

    model = DQN.load("Pong_Model")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rew, done, info = env.step(action)
        #env.render()
        if done:
            obs = env.reset()
        env.close()

if __name__ == "__main__":
        Pong_main()

