"""
CarRacing-v3 Reinforcement Learning Trainer using PPO

This script trains a reinforcement learning agent to play the CarRacing-v3 
environment from Gymnasium using Proximal Policy Optimization (PPO) algorithm
from stable-baselines3.

Environment Details:
- Observation: 96x96 RGB top-down view of the car and track
- Action Space: Discrete (5 actions) - do nothing, steer right, steer left, gas, brake
- Rewards: +1000/N per track tile visited, -0.1 per frame, -100 for going off-track
"""

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os


def make_env():
    """
    Create a single CarRacing-v3 environment with discrete action space.
    
    Returns:
        callable: A function that creates the environment when called
    """
    def _init():
        # Create the CarRacing environment with discrete actions
        env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
        return env
    return _init


def create_vectorized_env(n_envs=4):
    """
    Create vectorized environments for faster parallel training.
    
    Args:
        n_envs (int): Number of parallel environments to create
        
    Returns:
        VecFrameStack: Vectorized environment with frame stacking
    """
    # Create multiple parallel environments
    env_fns = [make_env() for _ in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)
    
    # Stack 4 frames together to give the agent temporal information
    # This helps the agent understand motion and velocity
    vec_env = VecFrameStack(vec_env, n_stack=4)
    
    return vec_env


def train_agent(total_timesteps=100000, n_envs=4, save_path="./models"):
    """
    Train a PPO agent on the CarRacing-v3 environment.
    
    Args:
        total_timesteps (int): Total number of timesteps to train for
        n_envs (int): Number of parallel environments
        save_path (str): Directory path to save the trained model
        
    Returns:
        PPO: The trained PPO model
    """
    print("=" * 60)
    print("Setting up CarRacing-v3 Training Environment")
    print("=" * 60)
    
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f"{save_path}/checkpoints", exist_ok=True)
    
    # Create vectorized training environment
    print(f"\nCreating {n_envs} parallel training environments...")
    train_env = create_vectorized_env(n_envs=n_envs)
    
    # Create a separate evaluation environment
    print("Creating evaluation environment...")
    eval_env = create_vectorized_env(n_envs=1)
    
    print("\nInitializing PPO agent with CNN policy...")
    print("Configuration:")
    print(f"  - Policy: CnnPolicy (for image observations)")
    print(f"  - Learning rate: 0.0003")
    print(f"  - Batch size: 64")
    print(f"  - Training timesteps: {total_timesteps}")
    
    # Initialize PPO agent with CNN policy for image-based observations
    model = PPO(
        policy="CnnPolicy",           # CNN policy for image inputs
        env=train_env,                # Vectorized environment
        learning_rate=0.0003,         # Learning rate for optimizer
        n_steps=128,                  # Number of steps to run for each environment per update
        batch_size=64,                # Minibatch size
        n_epochs=4,                   # Number of epochs when optimizing the surrogate loss
        gamma=0.99,                   # Discount factor
        gae_lambda=0.95,              # Factor for trade-off of bias vs variance for GAE
        clip_range=0.2,               # Clipping parameter for PPO
        verbose=1,                    # Print training information
        tensorboard_log=f"{save_path}/tensorboard_logs"  # TensorBoard logs
    )
    
    # Set up callbacks for periodic evaluation and model checkpointing
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=save_path,
        log_path=f"{save_path}/eval_logs",
        eval_freq=10000,              # Evaluate every 10000 steps
        n_eval_episodes=5,            # Use 5 episodes for evaluation
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=25000,              # Save model every 25000 steps
        save_path=f"{save_path}/checkpoints",
        name_prefix="ppo_carracing"
    )
    
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=True
    )
    
    # Save the final trained model
    final_model_path = f"{save_path}/ppo_carracing_final"
    model.save(final_model_path)
    print(f"\n✓ Training complete! Final model saved to: {final_model_path}")
    
    # Clean up environments
    train_env.close()
    eval_env.close()
    
    return model


def evaluate_agent(model, n_eval_episodes=10):
    """
    Evaluate the trained agent's performance.
    
    Args:
        model (PPO): The trained PPO model to evaluate
        n_eval_episodes (int): Number of episodes to evaluate over
        
    Returns:
        tuple: Mean reward and standard deviation of rewards
    """
    print("\n" + "=" * 60)
    print("Evaluating Agent Performance")
    print("=" * 60)
    
    # Create a fresh evaluation environment
    eval_env = create_vectorized_env(n_envs=1)
    
    print(f"\nRunning {n_eval_episodes} evaluation episodes...")
    
    # Evaluate the agent
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False
    )
    
    # Print evaluation results
    print("\n" + "-" * 60)
    print("Evaluation Results:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print("-" * 60)
    
    # Clean up
    eval_env.close()
    
    return mean_reward, std_reward


def load_and_test_agent(model_path, render=False, n_episodes=3):
    """
    Load a saved model and test it in the environment.
    
    Args:
        model_path (str): Path to the saved model
        render (bool): Whether to render the environment during testing
        n_episodes (int): Number of episodes to test
    """
    print(f"\nLoading model from: {model_path}")
    
    # Load the trained model
    model = PPO.load(model_path)
    
    # Create vectorized environment for testing (with frame stacking)
    # This ensures consistency with how the model was trained
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
    
    # Run test episodes
    for episode in range(n_episodes):
        obs = test_env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        while not done:
            # Get action from the model
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done_array, info = test_env.step(action)
            done = done_array[0]
            episode_reward += reward[0]
            steps += 1
        
        print(f"  Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {steps}")
    
    test_env.close()


def main():
    """
    Main function to orchestrate the training and evaluation process.
    """
    print("\n" + "=" * 60)
    print("CarRacing-v3 RL Training with PPO")
    print("=" * 60)
    print("\nThis script will:")
    print("  1. Create vectorized CarRacing-v3 environments")
    print("  2. Train a PPO agent with CNN policy")
    print("  3. Save the trained model")
    print("  4. Evaluate the agent's performance")
    print("\n" + "=" * 60 + "\n")
    
    # Configuration
    TOTAL_TIMESTEPS = 100000  # Total training timesteps (minimum as required)
    N_ENVS = 4                # Number of parallel environments
    SAVE_PATH = "./models"    # Directory to save models
    N_EVAL_EPISODES = 10      # Number of episodes for final evaluation
    
    # Step 1: Train the agent
    trained_model = train_agent(
        total_timesteps=TOTAL_TIMESTEPS,
        n_envs=N_ENVS,
        save_path=SAVE_PATH
    )
    
    # Step 2: Evaluate the trained agent
    mean_reward, std_reward = evaluate_agent(
        model=trained_model,
        n_eval_episodes=N_EVAL_EPISODES
    )
    
    # Step 3: Print final summary
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
    
    # Ask user if they want to watch the agent play
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
