import gymnasium
import numpy as np
import manipulator_mujoco

# Cria o ambiente com renderização
env = gymnasium.make('manipulator_mujoco/UR5eEnv-v0', render_mode='human')
obs, info = env.reset(seed=42)

# Parâmetros da trajetória circular (ajustados)
radius = 0.1                          # raio menor
center = np.array([0.5, 0.0])          # centro da trajetória (XY)
z = 0.3                                # altura constante
quat = np.array([0, 0, 0, 1])          # orientação fixa (sem rotação)
freq = 0.05                             # frequência mais lenta (0.1 Hz → 1 volta em 10s)
dt = 1 / 60                            # ~60 FPS

t = 0.0
for _ in range(10000):
    # Calcula posição do alvo na trajetória circular
    x = center[0] + radius * np.cos(2 * np.pi * freq * t)
    y = center[1] + radius * np.sin(2 * np.pi * freq * t)
    pose = np.concatenate([[x, y, z], quat])

    # Atualiza o alvo do OSC
    env.unwrapped.set_target_pose(pose)

    # Executa um passo da simulação
    obs, reward, terminated, truncated, info = env.step(np.zeros(6))

    if terminated or truncated:
        obs, info = env.reset()
        t = 0.0

    t += dt

env.close()
