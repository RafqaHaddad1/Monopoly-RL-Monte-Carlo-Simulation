🧠🎲 Reinforcement Learning + Monte Carlo Simulation on Monopoly
Project Overview
This project explores how Reinforcement Learning (RL) combined with Monte Carlo Simulation can be applied to the classic board game Monopoly. The goal is to enable an AI agent to learn optimal strategies through trial and error — without explicitly programming every possible move.

Developed under the supervision of Dr. Rachad Atat as part of my Spring 2025 academic term, this project showcases how parallel computing can accelerate training and learning processes in complex, stochastic environments like Monopoly.

💡 Why Monopoly?
Monopoly is:

Complex enough to require strategic decision-making

Rich in probabilistic outcomes (dice rolls, Chance and Community Chest cards)

Fun and relatable, making it an engaging platform for AI experimentation

🧠 How It Works
In this implementation:

The agent interacts with the game environment, making decisions at each step (e.g., buy/sell property, build houses).

Rewards are given based on outcomes (e.g., gaining money, acquiring properties, avoiding bankruptcy).

Using Monte Carlo methods, the agent simulates thousands of games to estimate the value of its actions over time.

This allows the agent to learn a policy that improves its chance of winning — all without a full mathematical model of the entire game.

⚙️ Implementation Details
🐍 Python
Initial implementation of the game logic and reinforcement learning algorithm.

Used to prototype and validate the agent's learning behavior.

💻 C + CUDA C
Translated the Python code to C for performance.

Parallelized simulations using CUDA C, enabling hundreds or thousands of games to run simultaneously on the GPU.

This massively reduces training time and allows better exploration of the game state space.

🎮 Game Features
Implemented features in the simplified Monopoly version include:

Game board with key actions (property purchase, rent collection, etc.)

Chance and Community Chest cards

Jail handling mechanics

Bankruptcy rules

Buying and selling properties

House construction

🚀 Future Improvements
Add trading between players

Incorporate more advanced RL algorithms (e.g., Deep Q-Learning)

Visualize agent behavior and policy evolution

Extend to full Monopoly ruleset
