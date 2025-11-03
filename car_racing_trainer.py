import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os


def make_env():
    def _init():
        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        return env
    return _init


def create_vectorized_env(n_envs=4):
    env_fns = [make_env() for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=4)
    return vec_env


def train_agent(total_timesteps=100000, n_envs=4, save_path="./models"):
    print("=" * 60)
    print("Setting up CarRacing-v3 Training Environment")
    print("=" * 60)
    
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/checkpoints", exist_ok=True)
    
    print(f"\nCreating {n_envs} parallel training environments...")
    train_env = create_vectorized_env(n_envs=n_envs)
    
    print("Creating evaluation environment...")
    eval_env = create_vectorized_env(n_envs=1)
    
    print("\nInitializing PPO agent with CNN policy...")
    print("Configuration:")
    print(f"  - Policy: CnnPolicy (for image observations)")
    print(f"  - Learning rate: 0.0003")
    print(f"  - Batch size: 64")
    print(f"  - Training timesteps: {total_timesteps}")
    
    model = PPO(
        policy="CnnPolicy",
        env=train_env,
        learning_rate=0.0003,
        n_steps=128,
        batch_size=64,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log=f"{save_path}/tensorboard_logs"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=f"{save_path}/eval_logs",
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,
        save_path=f"{save_path}/checkpoints",
        name_prefix="ppo_carracing"
    )
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    final_model_path = f"{save_path}/ppo_carracing_final"
    model.save(final_model_path)
    print(f"\n✓ Training complete! Final model saved to: {final_model_path}")
    
    train_env.close()
    eval_env.close()
    
    return model


def evaluate_agent(model, n_eval_episodes=10):
    print("\n" + "=" * 60)
    print("Evaluating Agent Performance")
    print("=" * 60)
    
    eval_env = create_vectorized_env(n_envs=1)
    
    print(f"\nRunning {n_eval_episodes} evaluation episodes...")
    
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    print("\n" + "-" * 60)
    print("Evaluation Results:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("-" * 60)
    
    eval_env.close()
    
    return mean_reward, std_reward


def load_and_test_agent(model_path, render=False, n_episodes=3):
    print(f"\nLoading model from: {model_path}")
    
    model = PPO.load(model_path)
    
    render_mode = "human" if render else "rgb_array"
    
    def make_test_env():
        def _init():
            return gym.make("CarRacing-v3", continuous=False, render_mode=render_mode)
        return _init
    
    test_env = DummyVecEnv([make_test_env()])
    test_env = VecFrameStack(test_env, n_stack=4)
    
    print(f"Testing agent for {n_episodes} episodes...")
    if render:
        print("🎮 Rendering enabled - watch the agent play!\n")
    
    for episode in range(n_episodes):
        obs = test_env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done_array, info = test_env.step(action)
            done = done_array[0]
            episode_reward += reward[0]
            steps += 1
        
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    test_env.close()


def main():
    print("\n" + "=" * 60)
    print("CarRacing-v3 RL Training with PPO")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Create vectorized CarRacing-v3 environments")
    print("  2. Train a PPO agent with CNN policy")
    print("  3. Save the trained model")
    print("  4. Evaluate the agent's performance")
    print("\n" + "=" * 60 + "\n")
    
    TOTAL_TIMESTEPS = 100000
    N_ENVS = 4
    SAVE_PATH = "./models"
    N_EVAL_EPISODES = 10
    
    trained_model = train_agent(
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS,
        save_path=SAVE_PATH
    )
    
    mean_reward, std_reward = evaluate_agent(
        model=trained_model,
        n_eval_episodes=N_EVAL_EPISODES
    )
    
    print("\n" + "=" * 60)
    print("Training Session Complete!")
    print("=" * 60)
    print(f"\nFinal Performance:")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Std Deviation: {std_reward:.2f}")
    print(f"\nModels saved in: {SAVE_PATH}")
    print(f"  - Best model: {SAVE_PATH}/best_model.zip")
    print(f"  - Final model: {SAVE_PATH}/ppo_carracing_final.zip")
    print("\n" + "=" * 60)
    
    print("\n🎮 Would you like to watch the trained agent play? (y/n): ", end="")
    try:
        response = input().strip().lower()
        if response in ['y', 'yes']:
            print("\n🚗 Launching visualization (close window to stop)...\n")
            load_and_test_agent(f"{SAVE_PATH}/best_model.zip", render=True)
        else:
            print("\n💡 You can watch the agent anytime by running:")
            print(f"   python watch_agent.py")
    except (EOFError, KeyboardInterrupt):
        print("\n💡 You can watch the agent anytime by running:")
        print(f"   python watch_agent.py")


if __name__ == "__main__":
    main()
