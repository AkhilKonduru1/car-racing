"""
Configuration file for CarRacing-v3 RL Training
Modify these parameters to customize your training setup
"""

# Training Configuration
TRAINING_CONFIG = {
    # Total number of timesteps to train (minimum 100,000 as required)
    "total_timesteps": 100000,
    
    # Number of parallel environments for faster training
    # Increase for faster training (requires more RAM)
    # Decrease if you encounter memory issues
    "n_envs": 4,
    
    # Directory to save trained models and logs
    "save_path": "./models",
    
    # Number of episodes for final evaluation
    "n_eval_episodes": 10,
}

# PPO Hyperparameters
PPO_CONFIG = {
    # Learning rate for the optimizer
    "learning_rate": 0.0003,
    
    # Number of steps to run for each environment per update
    # Higher values = more stable but slower updates
    "n_steps": 128,
    
    # Minibatch size for training
    # Must be a factor of (n_steps * n_envs)
    "batch_size": 64,
    
    # Number of epochs when optimizing the surrogate loss
    "n_epochs": 4,
    
    # Discount factor (gamma)
    # Higher values = agent considers long-term rewards more
    "gamma": 0.99,
    
    # Factor for trade-off of bias vs variance for GAE
    "gae_lambda": 0.95,
    
    # Clipping parameter for PPO
    "clip_range": 0.2,
    
    # Entropy coefficient for exploration
    # Higher values = more exploration
    "ent_coef": 0.01,
    
    # Value function coefficient
    "vf_coef": 0.5,
    
    # Maximum gradient norm for gradient clipping
    "max_grad_norm": 0.5,
    
    # Verbosity level (0: no output, 1: info, 2: debug)
    "verbose": 1,
}

# Evaluation Configuration
EVAL_CONFIG = {
    # Evaluate every N steps during training
    "eval_freq": 10000,
    
    # Number of episodes for periodic evaluation
    "n_eval_episodes": 5,
    
    # Use deterministic actions during evaluation
    "deterministic": True,
}

# Checkpoint Configuration
CHECKPOINT_CONFIG = {
    # Save model checkpoint every N steps
    "save_freq": 25000,
    
    # Prefix for checkpoint filenames
    "name_prefix": "ppo_carracing",
}

# Environment Configuration
ENV_CONFIG = {
    # Number of frames to stack together
    # Gives the agent temporal information about motion
    "n_stack": 4,
    
    # Render mode for environment ("rgb_array" or "human")
    # Use "human" to visualize during training (slower)
    "render_mode": "rgb_array",
    
    # Percentage of track that must be completed (0.0 to 1.0)
    "lap_complete_percent": 0.95,
    
    # Whether to randomize domain (colors) on each reset
    "domain_randomize": False,
}

# Advanced Options
ADVANCED_CONFIG = {
    # Use GPU if available (requires CUDA-enabled PyTorch)
    "use_gpu": True,
    
    # Random seed for reproducibility (None for random)
    "seed": None,
    
    # Enable TensorBoard logging
    "enable_tensorboard": True,
    
    # Policy network architecture (for custom CNN)
    # None = use default stable-baselines3 CNN
    "policy_kwargs": None,
    # Example custom architecture:
    # "policy_kwargs": {
    #     "features_extractor_kwargs": {
    #         "features_dim": 256
    #     }
    # }
}


def get_config():
    """
    Returns the complete configuration dictionary.
    """
    return {
        "training": TRAINING_CONFIG,
        "ppo": PPO_CONFIG,
        "eval": EVAL_CONFIG,
        "checkpoint": CHECKPOINT_CONFIG,
        "env": ENV_CONFIG,
        "advanced": ADVANCED_CONFIG,
    }


def print_config():
    """
    Prints the current configuration in a readable format.
    """
    config = get_config()
    
    print("\n" + "=" * 60)
    print("CarRacing-v3 Training Configuration")
    print("=" * 60)
    
    for section_name, section_config in config.items():
        print(f"\n{section_name.upper()}:")
        for key, value in section_config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    print_config()
