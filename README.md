# A-Critical-Review-of-DeepSeek-R1
# DeepSeek-R1 Review and a Toy Study of GRPO

A short critical review of [DeepSeek-R1](https://arxiv.org/abs/2501.12948) (DeepSeek-AI, 2025), plus a small simulation that compares the group-relative advantage used in GRPO with plain REINFORCE.

This is a learning project. It is a review and a toy experiment, not original research on R1 and not a replication of its results.

## Contents

| File | Description |
| --- | --- |
| `report.pdf` | The 7-page review, including the experiment write-up (Section 5) |
| `exp.py` | Policy-gradient simulation: REINFORCE, REINFORCE with a value baseline, and GRPO |
| `sweep.py` | Runs the learning-rate sweep over 30 seeds and produces the figure and results file |

## The experiment

**Question:** does GRPO's group-relative advantage help learning, beyond saving the memory of a critic model?

**Setup:** eight "prompts", each with one hidden correct answer, a sequence of three tokens from a vocabulary of six (216 possibilities). The policy is a table of softmax probabilities, not a neural network. The reward is binary: 1 for an exact match, 0 otherwise. Each update samples 16 answers per prompt.

**Methods compared:**

1. REINFORCE (Williams, 1992), with no baseline
2. REINFORCE with a running per-prompt value baseline, standing in for a critic
3. GRPO (Shao et al., 2024): reward minus the group mean, divided by the group standard deviation

**Metric:** updates until the 10-update average success rate first reaches 90%, capped at 600. Each configuration uses 30 seeds, each with new hidden answers, across 5 learning rates.

### Results

Updates to reach 90% success (mean ± SD over 30 seeds). ">600" means no run reached 90% within 600 updates.

| Learning rate | REINFORCE | REINFORCE + value baseline | GRPO |
| --- | --- | --- | --- |
| 0.1 | >600 | >600 | 475 ± 18 |
| 0.3 | 504 ± 26 | 503 ± 26 | 182 ± 17 |
| 1 | 176 ± 20 | 175 ± 20 | 76 ± 14 |
| 3 | 81 ± 16 | 81 ± 16 | 51 ± 16 |
| 10 | 51 ± 16 | 51 ± 16 | 45 ± 16 |

![Results](fig.png)

GRPO reached 90% success faster than both alternatives at every learning rate, and the gap shrank as the learning rate grew. The value baseline made no difference, which is expected because successes are rare and the running average stays near zero.

### How to read this

Dividing by the group standard deviation makes the advantage of a rare success large (about four times the raw reward when 1 of 16 answers is correct). That acts much like a larger learning rate. GRPO at a learning rate of 1 (76 updates) is close to REINFORCE at 3 (81 updates), so much of the speedup may come from rescaling and not from lower-variance gradients. I did **not** run the ablation that would separate the two (for example, mean-centering without dividing by the standard deviation), so this is a hypothesis.

## Limitations

- Toy setting: a lookup-table policy, a single-step task, a binary reward.
- No KL penalty, no clipping, no long chains of thought.
- Says nothing about whether reasoning emerges in real language models.
- Only one design choice is tested, in isolation.

## Reproducing

```bash
pip install -r requirements.txt
python sweep.py
```

This prints the results for each method and learning rate and writes `fig.png` and `sweep.json`. `exp.py` must stay in the same folder as `sweep.py`. Running `python exp.py` on its own repeats the per-method learning-rate tuning that I used while developing the experiment.

## AI assistance

The report text and the code were drafted with help from an AI assistant (Claude). I reviewed the report against the original paper and ran the code myself.

## References

- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. arXiv:2501.12948.
- Shao, Z. et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300.
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8(3–4), 229–256.

The full reference list is in the report.

## Author

Syed Ali Murtaza Zaidi · Murtaza.Zaidi@student.lab.fi · 2026

