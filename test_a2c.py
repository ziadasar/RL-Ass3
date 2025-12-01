import argparse
import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.A2C import A2CAgent


def evaluate(model_path, env_name, episodes=100, render=False):
    # ----------------------------------------------
    # Load environment
    # ----------------------------------------------
    env = gym.make(env_name, render_mode="human" if render else None)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # ----------------------------------------------
    # Re-create agent architecture (empty)
    # ----------------------------------------------
    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=(128, 128),
        device="cpu"
    )

    # ----------------------------------------------
    # Load weights from state_dict
    # ----------------------------------------------
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic.load_state_dict(checkpoint["critic"])

    print(f"\nLoaded trained A2C model from {model_path}\n")

    durations = []

    # ----------------------------------------------
    # Run test episodes
    # ----------------------------------------------
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits = agent.actor(obs_t)
            probs = torch.softmax(logits, dim=-1)
            action = torch.argmax(probs).item()

            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            obs = next_obs
            steps += 1

        durations.append(steps)
        print(f"Episode {ep+1}/{episodes}: {steps} steps")

    env.close()

    durations = np.array(durations)

    print("\n========== TEST SUMMARY ==========")
    print(f"Mean duration:  {durations.mean():.2f}")
    print(f"Std deviation: {durations.std():.2f}")
    print(f"Min duration:  {durations.min()}")
    print(f"Max duration:  {durations.max()}")
    print("=================================\n")

    # Save histogram
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=20, edgecolor='black')
    plt.title(f"A2C Test Episode Durations ({env_name})")
    plt.xlabel("Episode Length (steps)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{env_name}_a2c_test_hist.png")
    print(f"Saved histogram: {env_name}_a2c_test_hist.png")

    return durations


# ----------------------------------------------
# CLI
# ----------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")

    args = parser.parse_args()

    evaluate(
        model_path=args.model,
        env_name=args.env,
        episodes=args.episodes,
        render=args.render
    )
