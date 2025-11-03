"""
Watch Trained Agent Play CarRacing-v3

This script loads a trained model and visualizes it playing the CarRacing-v3 
environment in real-time. You can watch the AI agent drive around the track!

Usage:
    python watch_agent.py                           # Use default model path
    python watch_agent.py --model path/to/model.zip # Specify custom model
    python watch_agent.py --episodes 5              # Watch for 5 episodes
"""

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
