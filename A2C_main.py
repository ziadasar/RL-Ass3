import argparse
import gymnasium as gym
import torch
import wandb
from models.A2C import A2CAgent


def main():
    # --------------------------------------------------
    # PARSE CONSOLE ARGUMENTS
    # --------------------------------------------------
    parser = argparse.ArgumentParser(description="A2C Trainer")

    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--episodes", type=int, default=500, help="Training episodes")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--entropy", type=float, default=0.01, help="Entropy coefficient")

    args = parser.parse_args()

    # --------------------------------------------------
    # INIT W&B
    # --------------------------------------------------
    wandb.init(
        project="RL-Assignment3",
        name=f"A2C_{args.env}",
        config={
            "algorithm": "A2C",
            "environment": args.env,
            "episodes": args.episodes,
            "lr": args.lr,
            "gamma": args.gamma,
            "entropy_coef": args.entropy,
        }
    )

    # --------------------------------------------------
    # SETUP ENV + AGENT
    # --------------------------------------------------
    env = gym.make(args.env)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    agent = A2CAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        hidden_sizes=(128, 128),
        device="cpu"
    )

    # --------------------------------------------------
    # TRAINING LOOP
    # --------------------------------------------------
    print(f"\nTraining A2C on {args.env} for {args.episodes} episodes...\n")

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_reward = 0

        while not done:
            action, logp, value = agent.select_action(obs)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store(obs, action, reward, done, logp, value)

            obs = next_obs
            ep_reward += reward

        actor_loss, critic_loss = agent.update()

        # ------------------------------
        # LOG TO W&B
        # ------------------------------
        wandb.log({
            "episode": ep,
            "reward": ep_reward,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss
        })

        print(
            f"Episode {ep+1}/{args.episodes}  "
            f"Reward={ep_reward:.1f}  "
            f"A_Loss={actor_loss:.4f}  "
            f"C_Loss={critic_loss:.4f}"
        )

    # --------------------------------------------------
    # SAVE MODEL LOCALLY + LOG TO W&B
    # --------------------------------------------------
    model_path = f"a2c_{args.env}.pth"
    torch.save(agent, model_path)
    wandb.save(model_path)

    print(f"\nTraining complete! Model saved as {model_path}\n")

    env.close()
    wandb.finish()


if __name__ == "__main__":
    main()
