# Synthetic OJBench

This started after an onsite interview I didn't pass. The exercise involved generating synthetic tasks that were difficult enough for a model to fail on some attempts, but still within reach. I wanted to understand that process better, along with how the resulting data could be used for supervised fine-tuning (SFT) or reinforcement learning (RL).

I'm using this project to work through the whole process on a small scale: run a benchmark, generate useful training data, train a model, and see whether its score improves. Starting with a small model should make it easier to experiment locally and understand each step.

The starting point is [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), which I plan to run locally with [MLX](https://github.com/ml-explore/mlx-lm), and [OJBench](https://github.com/He-Ren/OJBench), a competitive programming benchmark. Programming problems give me a concrete way to check whether a solution is correct and meets execution limits.

The rough plan is to:

1. Get a local baseline and understand how benchmark evaluation works.
2. Generate new programming problems with tests and reference solutions, then measure how often the model can solve them.
3. Use that data for an SFT or RL training run. I haven't decided which approach to start with yet.
4. Evaluate again on OJBench to see whether the synthetic data helped on problems kept out of training.

The goal is to complete an experiment and understand the choices along the way. The plan will change as I learn what works.

**Current status:** initial scaffold and compatibility research. No baseline or training results yet.
