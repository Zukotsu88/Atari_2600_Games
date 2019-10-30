# Atari_2600_Games
The purpose of this project is to create working agents to win at several Atari 2600 games. 
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

In doing so it uses [``gym retro``](https://retro.readthedocs.io/en/latest/getting_started.html) in order to import the necessary Atari games. If you're having trouble importing the necessary ROMs, refer to [this](https://github.com/openai/retro/issues/60). 

Given that it's pixelated, this project uses ``stable_baseline``'s [CnnPolicy](https://stable-baselines.readthedocs.io/en/master/modules/dqn.html) with the Deep Q Learning algorithm.
