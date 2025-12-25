# 🧠🎲 Reinforcement Learning + Monte Carlo Simulation on Monopoly

## Overview

This project explores how **Reinforcement Learning (RL)** combined with **Monte Carlo Simulation** can be applied to the classic board game **Monopoly**. Developed as part of my Spring 2025 academic term, this project demonstrates how parallel computing techniques can accelerate training and learning in complex, stochastic environments.


## 🎯 Objective

To build an AI agent that learns optimal strategies in Monopoly through trial and error, without explicitly programming every move. The agent plays thousands of simulated games and learns from the outcomes to make better decisions over time.



## 🧠 How It Works

- The AI agent interacts with a simplified Monopoly environment.
- It receives **rewards** for favorable outcomes like acquiring property, earning money, or winning.
- Using **Monte Carlo methods**, the agent estimates long-term value of actions by averaging rewards over many episodes.
- This allows the agent to develop a **policy** that maximizes its chances of winning.
- No complete mathematical model of the game is required — the agent learns directly from experience.



## 🎮 Features

The game implementation includes:

- Simplified Monopoly board and game rules
- Buying/selling properties
- Building houses
- Chance and Community Chest cards
- Jail handling
- Bankruptcy mechanics



## ⚙️ Implementation Details

### 🐍 Python
- Used for the initial implementation of the game logic and RL agent.
- Helped prototype and test the learning approach.

### 💻 C and CUDA C
- Translated the Python codebase to C for performance optimization.
- Implemented **CUDA C** to parallelize simulations, allowing:
  - Hundreds or thousands of games to run in parallel
  - Drastic reduction in training time
  - Efficient exploration of state space



## 📢 Why Monopoly?

Because it's **fun** and **challenging** — a perfect playground for building smart, adaptive AI!

