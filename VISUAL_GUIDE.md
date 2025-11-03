# 🎮 Visual Testing Guide - Watch Your AI Drive!

## Quick Start - Watch Agent Play

After training your agent, you can watch it drive around the track in real-time!

### Method 1: Simple Command (Easiest! ⭐)

```bash
python watch_agent.py
```

That's it! A window will pop up showing your AI agent driving the car. The script automatically:
- Finds your best trained model
- Opens a visual window
- Shows the agent driving for 3 episodes
- Prints statistics for each episode

### Method 2: After Training

When you run the training script:

```bash
python car_racing_trainer.py
```

At the end, you'll see:
```
🎮 Would you like to watch the trained agent play? (y/n):
```

Just type `y` and press Enter to immediately watch your newly trained agent!

---

## Advanced Options

### Watch More Episodes

```bash
python watch_agent.py --episodes 10
```

### Use a Specific Model

```bash
# Use the final model
python watch_agent.py --model ./models/ppo_carracing_final.zip

# Use a checkpoint
python watch_agent.py --model ./models/checkpoints/ppo_carracing_50000_steps.zip
```

### Control Playback Speed

```bash
# Slow motion (good for analyzing behavior)
python watch_agent.py --delay 0.1

# Normal speed
python watch_agent.py --delay 0.01

# Fast (no delay)
python watch_agent.py --delay 0
```

### Combine Options

```bash
python watch_agent.py --episodes 5 --model ./models/best_model.zip --delay 0.05
```

---

## What You'll See

When the visual window opens, you'll see:

1. **The Race Track** - A randomly generated circuit
2. **Your AI Car** - The agent navigating the track
3. **Dashboard Indicators** at the bottom:
   - Speed
   - ABS sensors (4 indicators)
   - Steering wheel position
   - Gyroscope

4. **Terminal Output** showing:
   ```
   Episode 1/3
   ────────────────────────────────────────────────────────
   
   📊 Episode 1 Results:
      Total Reward: 789.34
      Steps Taken: 543
   ```

---

## Understanding Performance

### Good Performance
- **Reward**: 700-900+ (completes most/all of the track)
- **Behavior**: Smooth driving, follows track, handles turns well

### Medium Performance
- **Reward**: 300-700 (completes some of the track)
- **Behavior**: Stays on track but may be slow or make mistakes

### Poor Performance (Needs More Training)
- **Reward**: Below 300 or negative
- **Behavior**: Goes off track, crashes, or gets stuck

### Tips for Better Performance
1. Train longer: Use 500,000 - 1,000,000 timesteps
2. Monitor TensorBoard to ensure learning is happening
3. Try different hyperparameters in `config.py`

---

## Troubleshooting

### "No trained models found!"

**Solution:** Train a model first:
```bash
python car_racing_trainer.py
```

### Window doesn't appear

**Possible causes:**
1. **No display available** (SSH without X11):
   - Run on a machine with a display
   - Use X11 forwarding: `ssh -X user@host`
   
2. **Pygame/rendering issues**:
   ```bash
   pip install --upgrade gymnasium[box2d] pygame
   ```

### Agent performs poorly

**Solutions:**
1. Make sure you're using the best model:
   ```bash
   python watch_agent.py --model ./models/best_model.zip
   ```

2. Train longer - 100k steps is just a start!

3. Check training logs with TensorBoard

---

## Command Reference

```bash
# Basic usage
python watch_agent.py

# Show help and all options
python watch_agent.py --help

# Custom configuration
python watch_agent.py \
    --model ./models/best_model.zip \
    --episodes 5 \
    --delay 0.02
```

### Available Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | auto | Path to model (.zip file) |
| `--episodes` | int | 3 | Number of episodes to watch |
| `--delay` | float | 0.01 | Delay between frames (seconds) |

---

## Recording Your Agent

Want to save a video of your agent? You can use screen recording software:

**macOS:** Cmd+Shift+5 (built-in screen recorder)
**Windows:** Xbox Game Bar (Win+G)
**Linux:** OBS Studio or SimpleScreenRecorder

Or programmatically with `gymnasium`:
```python
import gymnasium as gym

env = gym.make("CarRacing-v3", render_mode="rgb_array")
env = gym.wrappers.RecordVideo(env, "./videos", episode_trigger=lambda x: True)
```

---

## Next Steps

1. ✅ Train your agent: `python car_racing_trainer.py`
2. 🎮 Watch it play: `python watch_agent.py`
3. 📊 Check TensorBoard logs to understand learning
4. 🔧 Tune hyperparameters for better performance
5. 🚀 Train for longer (500k-1M steps) for expert-level driving!

---

**Enjoy watching your AI learn to race! 🏎️💨**
