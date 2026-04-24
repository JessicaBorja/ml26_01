import gymnasium as gym
import wandb
import numpy as np

wandb.init(
    project="car-racing-dqn",
    name="entrega-intermedia",
    config={
        "episodios": 5,
        "max_steps_por_episodio": 500,
    }
)

env = gym.make("CarRacing-v3", render_mode="human")

print("Ambiente CarRacing iniciado")

for episodio in range(5):
    obs, info = env.reset()
    reward_total = 0
    done = False
    steps = 0

    while not done and steps < 500:
        accion = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(accion)
        done = terminated or truncated
        reward_total += reward
        steps += 1

        wandb.log({"reward_step": reward, "episodio": episodio})

    wandb.log({"reward_total": reward_total, "steps": steps})
    print(f"Episodio {episodio+1}/5 | Reward: {reward_total:.1f}")

env.close()
wandb.finish()