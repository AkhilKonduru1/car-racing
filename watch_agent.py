import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import argparse
import os
import time


def find_best_model(models_dir="./models"):
    best_model_path = f"{models_dir}/best_model.zip"
    if os.path.exists(best_model_path):
        return best_model_path
    
    final_model_path = f"{models_dir}/ppo_carracing_final.zip"
    if os.path.exists(final_model_path):
        return final_model_path
    
    checkpoints_dir = f"{models_dir}/checkpoints"
    if os.path.exists(checkpoints_dir):
        checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith('.zip')]
        if checkpoint_files:
            checkpoint_files.sort(reverse=True)
            return os.path.join(checkpoints_dir, checkpoint_files[0])
    
    return None


def watch_agent(model_path, n_episodes=5, delay=0.01):
    print(f"\n{'='*60}")
    print("CarRacing-v3 Agent Visualization")
    print(f"{'='*60}\n")
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file not found at {model_path}")
        print("\nMake sure you have trained a model first by running:")
        print("  python car_racing_trainer.py")
        return
    
    print(f"Loading model from: {model_path}\n")
    model = PPO.load(model_path)
    
    def make_env():
        def _init():
            return gym.make("CarRacing-v3", continuous=False, render_mode="human")
        return _init
    
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)  
    
    print(f"🎮 Starting visualization for {n_episodes} episodes...")
    print("🎬 Close the window to stop early\n")
    
    try:
        for episode in range(n_episodes):
            obs = env.reset()
            episode_reward = 0
            done = False
            steps = 0
            
            print(f"{'─'*60}")
            print(f"Episode {episode + 1}/{n_episodes}")
            print(f"{'─'*60}")
            
            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done_array, info = env.step(action)
                done = done_array[0]
                episode_reward += reward[0]
                steps += 1
                
                if delay > 0:
                    time.sleep(delay)
            
            print(f"Reward: {episode_reward:.2f} | Steps: {steps}")
            print()  
        
        print(f"{'='*60}")
        print("Visualization Complete!")
        print(f"{'='*60}\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Visualization interrupted by user")
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Watch a trained CarRacing-v3 agent play"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained model (default: auto-detect best model)"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to watch (default: 5)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay between frames in seconds (default: 0.01)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory containing saved models (default: ./models)"
    )
    
    args = parser.parse_args()
    
    if args.model is None:
        print("Auto-detecting best available model...")
        model_path = find_best_model(args.models_dir)
        if model_path is None:
            print(f"\n❌ Error: No trained models found in {args.models_dir}")
            print("\nTrain a model first by running:")
            print("  python car_racing_trainer.py")
            return
        print(f"Found: {model_path}\n")
    else:
        model_path = args.model
    
    watch_agent(
        model_path=model_path,
        n_episodes=args.episodes,
        delay=args.delay
    )


if __name__ == "__main__":
    main()

import gymnasium as gym
from stable_baselines3 import PPO
import argparse
import os
import time


def watch_agent(model_path, num_episodes=3, delay=0.01):
    """
    Load and visualize a trained agent playing CarRacing-v3.
    
    Args:
        model_path (str): Path to the saved model (.zip file)
        num_episodes (int): Number of episodes to watch
        delay (float): Delay between frames (seconds) to control speed
    """
    print("\n" + "=" * 60)
    print("CarRacing-v3 - Watch Trained Agent")
    print("=" * 60)
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"\n❌ Error: Model not found at {model_path}")
        print("\nAvailable models in ./models/:")
        if os.path.exists("./models"):
            for file in os.listdir("./models"):
                if file.endswith(".zip"):
                    print(f"  - {file}")
        else:
            print("  No models directory found. Train a model first!")
        return
    
    print(f"\n📁 Loading model from: {model_path}")
    
    try:
        # Load the trained model
        model = PPO.load(model_path)
        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create the environment with human rendering
    print("\n🎮 Creating environment with visual rendering...")
    env = gym.make("CarRacing-v3", continuous=False, render_mode="human")
    print("✓ Environment created!")
    
    print(f"\n🚗 Starting visualization for {num_episodes} episode(s)...")
    print("=" * 60)
    print("\n💡 Tip: Close the render window to stop early\n")
    
    # Run episodes
    for episode in range(num_episodes):
        print(f"\n{'─' * 60}")
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"{'─' * 60}")
        
        obs, info = env.reset()
        episode_reward = 0
        step_count = 0
        done = False
        
        # Run until episode is done
        while not done:
            try:
                # Get action from the trained model
                action, _states = model.predict(obs, deterministic=True)
                
                # Take action in environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                step_count += 1
                
                # Optional: Add small delay to control playback speed
                if delay > 0:
                    time.sleep(delay)
                
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user")
                env.close()
                return
        
        # Print episode statistics
        print(f"\n📊 Episode {episode + 1} Results:")
        print(f"   Total Reward: {episode_reward:.2f}")
        print(f"   Steps Taken: {step_count}")
        
        # Brief pause between episodes
        if episode < num_episodes - 1:
            print("\n⏳ Starting next episode in 2 seconds...")
            time.sleep(2)
    
    # Clean up
    env.close()
    
    print("\n" + "=" * 60)
    print("✓ Visualization complete!")
    print("=" * 60 + "\n")


def find_best_model():
    """
    Automatically find the best available model.
    
    Returns:
        str: Path to the best model, or None if not found
    """
    # Priority order for model selection
    model_paths = [
        "./models/best_model.zip",           # Best performing model
        "./models/ppo_carracing_final.zip",  # Final trained model
    ]
    
    # Check priority models
    for path in model_paths:
        if os.path.exists(path):
            return path
    
    # Check for any checkpoint models
    if os.path.exists("./models/checkpoints"):
        checkpoints = [f for f in os.listdir("./models/checkpoints") if f.endswith(".zip")]
        if checkpoints:
            # Sort and return the latest checkpoint
            checkpoints.sort()
            return os.path.join("./models/checkpoints", checkpoints[-1])
    
    return None


def main():
    """
    Main function with command-line argument parsing.
    """
    parser = argparse.ArgumentParser(
        description="Watch a trained PPO agent play CarRacing-v3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python watch_agent.py
  python watch_agent.py --model ./models/best_model.zip
  python watch_agent.py --episodes 5 --delay 0.02
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to the trained model (.zip file). If not specified, uses best available model."
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to watch (default: 3)"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=0.01,
        help="Delay between frames in seconds (default: 0.01). Increase to slow down playback."
    )
    
    args = parser.parse_args()
    
    # Determine which model to use
    if args.model:
        model_path = args.model
    else:
        print("🔍 No model specified, searching for best available model...")
        model_path = find_best_model()
        
        if model_path is None:
            print("\n❌ No trained models found!")
            print("\nPlease train a model first by running:")
            print("  python car_racing_trainer.py")
            print("\nOr specify a model path:")
            print("  python watch_agent.py --model path/to/model.zip")
            return
        else:
            print(f"✓ Found model: {model_path}")
    
    # Watch the agent play
    watch_agent(
        model_path=model_path,
        num_episodes=args.episodes,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
