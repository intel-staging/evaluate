---
title: Adversarial Glue
emoji: 👀
colorFrom: pink
colorTo: green
sdk: static
pinned: false
license: apache-2.0
---

# Bias AUC Metrics

## Description

This evaluation metric computes a a suite of threshold-agnostic metrics that provide a nuanced view of this unintended bias, by considering the various ways that a classifier’s score distribution can vary across designated groups. This contians three metrics:

* Subgroup AUC: Calculates AUC on only the examples from the subgroup. This represents model understanding and separability within the subgroup itself.

* Background Positive Subgroup Negative (BPSN) AUC: Calculates AUC on the positive examples from the background and the negative examples from the subgroup. This value would be reduced when scores for negative examples in the subgroup are higher than scores for other positive
examples. These examples would likely appear as false positives within the subgroup at
many thresholds.

* Background Negative Subgroup Positive (BNSP) AUC: Calculates AUC on the negative examples from the background and the positive examples from the subgroup. This value would be reduced when scores for positive examples in the subgroup are lower than scores for other negative examples. The examples would likely appear as false negatives within the subgroup at many thresholds.

## How to use

After installation, there are two steps: (1) loading the metric; and (2) calculating the metric.

1. **Loading the relevant GLUE metric** : This suite loads an evaluation metric and computes the following bias metrics: Subgroup,  BPSN, and BNSP

2. **Calculating the metric**: the metric takes three inputs: the name of the model or pipeline


```python
from evaluate import EvaluationSuite

from evaluate import load
bias_auc_metric = load('Intel/bias_auc')

targets = [['Islam'],
 ['None'],
 ['None'],
 ['Islam']]

label = [0,
 0,
 1,
 1]

output = [[0.44, 0.56],
 [0.43, 0.57],
 [0.40, 0.60],
 [0.38, 0.62]]

subgroups = set(group for group_list in a for group in group_list)

bias_auc_metric.compute(
    target=target,
    label=label,
    output=output,
    subgroups=subgroups)
                 
>>> {'Islam': {'Subgroup': 1.0, 'BPSN': 1.0, 'BNSP': 1.0},
 'None': {'Subgroup': 1.0, 'BPSN': 1.0, 'BNSP': 1.0},
 'Overall': {'Subgroup generalized mean': 1.0,
  'BPSN generalized mean': 1.0,
  'BNSP generalized mean': 1.0,
  'Overall AUC': 1.0}}

```

## Output results

Each subgroup will be a key with a dictionary object as a value, which contains the bias metric results for that subgroup. A key for "Overall" is also added with containing  the generalized mean for each of the Bias AUCs.

### Values from popular papers

See the [Nuanced Metrics for Measuring Unintended Bias
with Real Data for Text Classification](https://arxiv.org/pdf/1903.04561.pdf) for reports on results on synthetic data. Also see, [HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection](https://arxiv.org/pdf/2012.10289.pdf) for results on HateXplain dataset.

## Citation

```bibtex
@inproceedings{borkan2019nuanced,
  title={Nuanced metrics for measuring unintended bias with real data for text classification},
  author={Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain, Nithum and Vasserman, Lucy},
  booktitle={Companion proceedings of the 2019 world wide web conference},
  pages={491--500},
  year={2019}
}
```

```bibtex
@inproceedings{mathew2021hatexplain,
  title={Hatexplain: A benchmark dataset for explainable hate speech detection},
  author={Mathew, Binny and Saha, Punyajoy and Yimam, Seid Muhie and Biemann, Chris and Goyal, Pawan and Mukherjee, Animesh},
  booktitle={Proceedings of the AAAI conference on artificial intelligence},
  volume={35},
  number={17},
  pages={14867--14875},
  year={2021}
}
```S