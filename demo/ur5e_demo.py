import gymnasium
import manipulator_mujoco
import numpy as np

# Cria o ambiente com renderização
env = gymnasium.make('manipulator_mujoco/UR5eEnv-v0', render_mode='human')  # ambiente com JointEffortController

# Reseta o ambiente
obs, info = env.reset(seed=42)

# Loop de simulação
while True:
    # Envia uma ação aleatória (esforços nas 6 juntas)
    action = env.action_space.sample()

    # Executa um passo com a ação
    obs, reward, terminated, truncated, info = env.step(action)

    # Reinicia se o episódio terminar
    if terminated or truncated:
        obs, info = env.reset()

env.close()
