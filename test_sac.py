import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.SAC import SACAgent
from utils.discrete_pendulum import DiscretePendulum
import os

# ============================================================
# ENV MENU
# ============================================================
ENV_MENU = {
    1: ("CartPole-v1", True, None),
    2: ("MountainCar-v0", True, None),
    3: ("Acrobot-v1", True, None),
    4: ("Pendulum-v1", False, 9)  # continuous but discretized
}

# ============================================================
# EVALUATE FUNCTION
# ============================================================
def evaluate(model_path, env_name, act_dim, discrete, episodes, action_bins=None):
    if env_name == "Pendulum-v1" and not discrete:
        env = DiscretePendulum(gym.make(env_name), num_actions=action_bins)
    else:
        env = gym.make(env_name)

    obs_dim = env.observation_space.shape[0]

    # Load checkpoint with weights_only=False
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    agent = SACAgent(
        state_dim=obs_dim,
        action_dim=act_dim,
        hidden_dim=checkpoint["hidden_dim"],
        gamma=checkpoint["gamma"],
        actor_lr=checkpoint["actor_lr"],
        critic1_lr=checkpoint["critic1_lr"],
        critic2_lr=checkpoint["critic2_lr"],
        entropy_coef=checkpoint["entropy_coef"]
    )

    agent.actor.load_state_dict(checkpoint["actor"])
    agent.critic1.load_state_dict(checkpoint["critic1"])
    agent.critic2.load_state_dict(checkpoint["critic2"])

    print(f"\nLoaded model: {model_path}")
    print(f"Environment: {env_name}")
    print(f"Discrete: {discrete}   |   Action dim: {act_dim}\n")

    durations = []

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0

        while not done:
            action = agent.select_action(obs)
            next_obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            obs = next_obs
            steps += 1

        durations.append(steps)
        print(f"Episode {ep+1}/{episodes}: {steps} steps")

    env.close()

    durations = np.array(durations)

    # Log results to a text file
    os.makedirs("tests/SAC", exist_ok=True)
    log_file = os.path.join("tests/SAC", f"{env_name}_sac_test_results.txt")
    with open(log_file, "w") as f:
        f.write("Episode Rewards:\n")
        for i, reward in enumerate(durations, 1):
            f.write(f"Episode {i}: {reward} steps\n")
        f.write("\n========== TEST SUMMARY ============\n")
        f.write(f"Mean duration:  {durations.mean():.2f}\n")
        f.write(f"Std deviation: {durations.std():.2f}\n")
        f.write(f"Min duration:  {durations.min()}\n")
        f.write(f"Max duration:  {durations.max()}\n")
        f.write("====================================\n")

    print(f"\nResults saved to: {log_file}")

    print("\n========== TEST SUMMARY ============")
    print(f"Mean duration:  {durations.mean():.2f}")
    print(f"Std deviation: {durations.std():.2f}")
    print(f"Min duration:  {durations.min()}")
    print(f"Max duration:  {durations.max()}")
    print("====================================\n")

    # Save histogram
    filename = f"{env_name}_sac_test_hist.png"
    plt.figure(figsize=(8, 5))
    plt.hist(durations, bins=20, edgecolor='black')
    plt.title(f"SAC Test Episode Durations ({env_name})")
    plt.xlabel("Episode Length")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.savefig(filename)

    print(f"Saved histogram: {filename}")


# ============================================================
# INTERACTIVE MODE (NO ARGUMENTS)
# ============================================================
if __name__ == "__main__":

    print("\n=== Choose environment to test ===")
    print("1 → CartPole-v1")
    print("2 → MountainCar-v0")
    print("3 → Acrobot-v1")
    print("4 → Pendulum-v1 (discrete)")
    print("=================================\n")

    choice = int(input("Enter choice (1–4): ").strip())

    if choice not in ENV_MENU:
        print("❌ Invalid choice.")
        exit()

    env_name, discrete, action_bins = ENV_MENU[choice]

    # Automatically construct the model path
    model_path = os.path.join("trained_models", "SAC", f"sac_{env_name}.pth")

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        exit()

    episodes = int(input("How many episodes to test? (default 100): ") or "100")

    evaluate(
        model_path=model_path,
        env_name=env_name,
        act_dim=action_bins if not discrete else gym.make(env_name).action_space.n,
        discrete=discrete,
        episodes=episodes,
        action_bins=action_bins
    )