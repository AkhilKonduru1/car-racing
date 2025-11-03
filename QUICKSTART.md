# Quick Start Guide - CarRacing-v3 RL Training

## 🚀 Quick Start (3 Steps)

### Step 1: Install Dependencies

```bash
# Option A: Use the setup script (recommended)
./setup.sh

# Option B: Manual installation
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Note**: On macOS, you may need to install SWIG first:
```bash
brew install swig
```

### Step 2: Start Training

```bash
python car_racing_trainer.py
```

The script will:
- Create 4 parallel CarRacing environments
- Train a PPO agent with CNN policy for 100,000 timesteps
- Save checkpoints every 25,000 steps
- Evaluate periodically and save the best model
- Print final evaluation results (mean ± std reward)
- **Ask if you want to watch the agent play!** 🎮

### Step 3: Watch Your Agent Drive! 🚗💨

After training (or anytime later), watch your AI agent drive:

```bash
python watch_agent.py
```

A window will pop up showing the agent navigating the track in real-time!

**More options:**
```bash
# Watch for 5 episodes
python watch_agent.py --episodes 5

# Use a specific model
python watch_agent.py --model ./models/ppo_carracing_final.zip

# Slow down playback
python watch_agent.py --delay 0.05
```

### Step 4: Monitor Training Progress (Optional)

While training, open a new terminal and run:

```bash
tensorboard --logdir=./models/tensorboard_logs
```

Then visit http://localhost:6006 to see real-time training metrics.

---

## 📊 What to Expect

**Training Output Example:**
```
==========================================================
CarRacing-v3 RL Training with PPO
==========================================================

Setting up CarRacing-v3 Training Environment
Creating 4 parallel training environments...
Creating evaluation environment...
Initializing PPO agent with CNN policy...

Starting Training...
[Progress bar and training logs...]

✓ Training complete! Final model saved to: ./models/ppo_carracing_final

Evaluating Agent Performance
Running 10 evaluation episodes...

Evaluation Results:
  Mean Reward: XXX.XX ± XX.XX
```

---

## 📁 Project Files

| File | Description |
|------|-------------|
| `car_racing_trainer.py` | Main training script with complete implementation |
| `watch_agent.py` | **Watch your trained agent play visually!** |
| `requirements.txt` | All Python dependencies |
| `config.py` | Advanced configuration options (optional) |
| `README.md` | Complete documentation |
| `setup.sh` | Automated setup script |
| `QUICKSTART.md` | This file |

---

## 🎮 Testing Your Trained Agent

After training, watch the agent drive visually:

**Simple Method (Recommended):**
```bash
python watch_agent.py
```

This loads your best model and shows the agent driving in a window!

**During Training:**
When training finishes, you'll see:
```
🎮 Would you like to watch the trained agent play? (y/n):
```
Type `y` to immediately see your agent in action!

**Command Line Options:**
```bash
# Watch for more episodes
python watch_agent.py --episodes 10

# Use a specific checkpoint
python watch_agent.py --model ./models/checkpoints/ppo_carracing_50000_steps.zip

# Slow motion playback
python watch_agent.py --delay 0.1
```

---

## ⚙️ Customization

### Change Training Duration

Edit in `car_racing_trainer.py`:
```python
TOTAL_TIMESTEPS = 500000  # Train for 500k steps instead
```

### Adjust Parallel Environments

```python
N_ENVS = 2  # Use fewer environments if you have memory issues
```

### Use the Config File

For advanced customization, modify `config.py` and import it:
```python
from config import get_config

config = get_config()
total_timesteps = config["training"]["total_timesteps"]
```

---

## 🐛 Common Issues

### Issue: "Import gymnasium could not be resolved"
**Solution**: Install dependencies first
```bash
pip install -r requirements.txt
```

### Issue: "Box2D not found"
**Solution**: Install SWIG first
```bash
# macOS
brew install swig

# Linux
sudo apt-get install swig

# Then reinstall
pip install gymnasium[box2d]
```

### Issue: "Out of memory"
**Solution**: Reduce parallel environments
```python
N_ENVS = 1  # Use just one environment
```

---

## 📈 Performance Tips

1. **Train Longer**: 100k steps is minimum. For better results, train 500k-1M steps
2. **Use GPU**: If available, PyTorch will automatically use CUDA
3. **Tune Hyperparameters**: Modify `config.py` for experimentation
4. **Monitor Progress**: Use TensorBoard to track learning curves

---

## 🎯 Next Steps

1. ✅ Run the basic training (100k steps)
2. 📊 Check TensorBoard metrics
3. 🎮 Test the trained agent visually
4. 🔧 Experiment with hyperparameters
5. 🚀 Train for longer (500k+ steps) for better performance

---

## 💡 Understanding the Code

The implementation includes:

- **Vectorized Environments**: 4 parallel environments for faster training
- **Frame Stacking**: 4 frames stacked to give temporal context
- **CNN Policy**: Convolutional neural network to process image observations
- **PPO Algorithm**: Proximal Policy Optimization for stable learning
- **Automatic Checkpointing**: Regular saves during training
- **Best Model Selection**: Automatically saves the best performing model
- **Comprehensive Evaluation**: Final testing over multiple episodes

---

## 📚 Resources

- [Gymnasium Docs](https://gymnasium.farama.org/)
- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Paper](https://arxiv.org/abs/1707.06347)

---

**Ready to train your racing AI? Run `python car_racing_trainer.py` and let's go! 🏎️💨**
