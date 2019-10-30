# Atari_2600_Games
The purpose of this reinforcement learning project is to create working agents in to win at several Atari 2600 games using a Deep Q Learning algorithm. 
This includes: 
1. AirRaid
2. Air-Striker Genesis
3. Assault
4. Asterix
5. Asteroids
6. Centipede
7. Defender
8. DemonAttack
9. Frostbite
10. Hero
11. IceHockey
12. MsPacman
13. Pong
14. Riverraid
15. Space-Invaders

![Space Invaders](https://thumbs.gfycat.com/CookedFriendlyAntarcticfurseal-size_restricted.gif)
![Asteroids](https://thumbs.gfycat.com/EachQuerulousHawk-size_restricted.gif)
![AirStriker Genesis](https://miro.medium.com/max/1280/1*tbYUjwhnuFdEiXV51qXCpQ.gif)

In doing so it uses [``gym retro``](https://retro.readthedocs.io/en/latest/getting_started.html) in order to import the necessary Atari games. If you're having trouble importing the necessary ROMs, refer to [this](https://github.com/openai/retro/issues/60). 

Given that it's pixelated, this project uses ``stable_baseline``'s [CnnPolicy](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html) with the Deep Q Learning algorithm in order to train the models.
