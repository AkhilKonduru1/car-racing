# CarRacing-v3 Reinforcement Learning Training

This project trains a reinforcement learning agent to play the CarRacing-v3 environment from Gymnasium using Proximal Policy Optimization (PPO) from stable-baselines3.

## Environment Details

- **Observation Space**: 96x96 RGB top-down view of the car and track
- **Action Space**: Discrete (5 actions)
  - 0: Do nothing
  - 1: Steer right
  - 2: Steer left
  - 3: Gas
  - 4: Brake
- **Rewards**: 
  - +1000/N for each track tile visited (N = total tiles)
  - -0.1 per frame
  - -100 for going off-track

## Features

✅ PPO algorithm with CNN policy for image-based observations  
✅ Vectorized environments for faster parallel training  
✅ Frame stacking (4 frames) for temporal information  
✅ Automatic model checkpointing during training  
✅ Periodic evaluation with best model saving  
✅ TensorBoard logging for monitoring training progress  
✅ Comprehensive evaluation with mean and std deviation of rewards  

## Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**Note**: The `gymnasium[box2d]` package requires Box2D, which may need additional system dependencies:

- **macOS**: `brew install swig`
- **Linux**: `sudo apt-get install swig`
- **Windows**: SWIG should be installed automatically via pip

## Usage

### Basic Training

Run the training script with default settings (100,000 timesteps):

```bash
python car_racing_trainer.py
```

After training completes, you'll be prompted if you want to watch the agent play!

### Watch Your Trained Agent Play 🎮

You can watch your trained agent drive around the track anytime:

```bash
# Watch with default settings (3 episodes, uses best model)
python watch_agent.py

# Watch for 5 episodes
python watch_agent.py --episodes 5

# Use a specific model
python watch_agent.py --model ./models/ppo_carracing_final.zip

# Slow down playback (increase delay between frames)
python watch_agent.py --delay 0.05
```

The render window will show the car driving in real-time!

### What Happens During Training

1. **Environment Setup**: Creates 4 parallel CarRacing environments for faster training
2. **Agent Initialization**: Initializes a PPO agent with CNN policy
3. **Training Loop**: Trains for 100,000+ timesteps with:
   - Evaluation every 10,000 steps
   - Model checkpoints every 25,000 steps
   - Best model automatically saved
4. **Final Evaluation**: Evaluates the agent over 10 episodes and prints mean ± std rewards

### Output Structure

After training, you'll find:

```
models/
├── best_model.zip              # Best performing model during training
├── ppo_carracing_final.zip     # Final trained model
├── checkpoints/                # Periodic checkpoints
│   ├── ppo_carracing_25000_steps.zip
│   ├── ppo_carracing_50000_steps.zip
│   └── ...
├── eval_logs/                  # Evaluation logs
└── tensorboard_logs/           # TensorBoard logs
```

### Monitor Training with TensorBoard

```bash
tensorboard --logdir=./models/tensorboard_logs
```

Then open http://localhost:6006 in your browser.

## Code Structure

The script is organized into modular functions:

- **`make_env()`**: Creates a single CarRacing environment
- **`create_vectorized_env()`**: Creates vectorized environments with frame stacking
- **`train_agent()`**: Main training loop with PPO
- **`evaluate_agent()`**: Evaluates agent performance over multiple episodes
- **`load_and_test_agent()`**: Loads and tests a saved model
- **`main()`**: Orchestrates the entire training process

## Customization

You can modify the configuration in the `main()` function:

```python
TOTAL_TIMESTEPS = 100000  # Increase for longer training
N_ENVS = 4                # Number of parallel environments
N_EVAL_EPISODES = 10      # Episodes for evaluation
```

Or modify PPO hyperparameters in `train_agent()`:

```python
model = PPO(
    policy="CnnPolicy",
    env=train_env,
    learning_rate=0.0003,   # Adjust learning rate
    n_steps=128,            # Steps per environment per update
    batch_size=64,          # Minibatch size
    n_epochs=4,             # Optimization epochs
    gamma=0.99,             # Discount factor
    # ... other parameters
)
```

## Testing a Trained Model

### Option 1: Using the Watch Script (Recommended)

The easiest way to visualize your trained agent:

```bash
python watch_agent.py
```

This automatically finds and loads your best model, then shows the agent driving!

### Option 2: In Python Code

```python
from stable_baselines3 import PPO
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Load the trained model
model = PPO.load("./models/best_model.zip")

# Create environment with frame stacking (same as training)
def make_env():
    def _init():
        return gym.make("CarRacing-v3", continuous=False, render_mode="human")
    return _init

env = DummyVecEnv([make_env()])
env = VecFrameStack(env, n_stack=4)

# Watch it play
obs = env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

env.close()
```

## Performance Expectations

- **Initial training**: The agent may perform poorly at first (negative rewards)
- **After 100k steps**: The agent should learn basic track following
- **For better results**: Train for 500k - 1M timesteps

## Troubleshooting

### Box2D Installation Issues

If you encounter errors installing `box2d-py`:

```bash
# macOS
brew install swig
pip install box2d-py

# Linux
sudo apt-get install swig
pip install box2d-py
```

### CUDA/GPU Support

To use GPU acceleration (if available):

```bash
# Install PyTorch with CUDA support first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Then install other requirements
pip install -r requirements.txt
```

### Memory Issues

If you run into memory issues, reduce the number of parallel environments:

```python
N_ENVS = 2  # or even 1
```

## References

- [Gymnasium CarRacing-v3 Documentation](https://gymnasium.farama.org/environments/box2d/car_racing/)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

## License

This project is for educational purposes.
