---
title: Adversarial Glue
emoji: 👀
colorFrom: pink
colorTo: green
sdk: static
pinned: false
license: apache-2.0
---

# Adversarial GLUE Evaluation Suite

## Description

This evaluation suite compares the GLUE results with Adversarial GLUE (AdvGLUE), a multi-task benchmark that evaluates modern large-scale language models robustness with respect to various types of adversarial attacks.

## How to use

This suite requires installations of the following fork [IntelAI/evaluate](https://github.com/IntelAI/evaluate/tree/develop).

After installation, there are two steps: (1) loading the Adversarial GLUE suite; and (2) calculating the metric.

1. **Loading the relevant GLUE metric** : This suite loads an evaluation suite subtasks for the following GLUE tasks: `sst2`,  `mnli`, `qnli`, `rte`, and `qqp`. It also runs these same t

More information about the different subsets of the GLUE dataset can be found on the [GLUE dataset page](https://huggingface.co/datasets/glue).

2. **Calculating the metric**: the metric takes one input: the name of the model or pipeline


```python
from evaluate import EvaluationSuite

suite = EvaluationSuite.load('intel/adversarial_glue')
mc_results,  = suite.run("gpt2")
```

## Output results

The output of the metric depends on the GLUE subset chosen, consisting of a dictionary that contains one or several of the following metrics:

`accuracy`: the proportion of correct predictions among the total number of cases processed, with a range between 0 and 1 (see [accuracy](https://huggingface.co/metrics/accuracy) for more information). 


### Values from popular papers

The [original GLUE paper](https://huggingface.co/datasets/glue) reported average scores ranging from 58 to 64%, depending on the model used (with all evaluation values scaled by 100 to make computing the average possible).

For more recent model performance, see the [dataset leaderboard](https://paperswithcode.com/dataset/glue).

## Examples 

For full example see [HF Evaluate Adversarial Attacks.ipynb](https://github.com/IntelAI/evaluate/blob/develop/notebooks/HF%20Evaluate%20Adversarial%20Attacks.ipynb)

## Limitations and bias
This metric works only with datasets that have the same format as the [GLUE dataset](https://huggingface.co/datasets/glue).

While the GLUE dataset is meant to represent "General Language Understanding", the tasks represented in it are not necessarily representative of language understanding, and should not be interpreted as such. 

## Citation

```bibtex
 @inproceedings{wang2021adversarial,
  title={Adversarial GLUE: A Multi-Task Benchmark for Robustness Evaluation of Language Models},
  author={Wang, Boxin and Xu, Chejian and Wang, Shuohang and Gan, Zhe and Cheng, Yu and Gao, Jianfeng and Awadallah, Ahmed Hassan and Li, Bo},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```

