import evaluate
import datasets
from datasets.features import Sequence, Value
from sklearn.metrics import roc_auc_score
import numpy as np


_KWARGS_DESCRIPTION = """\
Suite of threshold-agnostic metrics that provide a nuanced view
of this unintended bias, by considering the various ways that a
classifier's score distribution can vary across designated groups.
"""

_CITATION = """\
@inproceedings{borkan2019nuanced,
  title={Nuanced metrics for measuring unintended bias with real
  data for text classification},
  author={Borkan, Daniel and Dixon, Lucas and Sorensen, Jeffrey and Thain,
  Nithum and Vasserman, Lucy},
  booktitle={Companion proceedings of the 2019 world wide web conference},
  pages={491--500},
  year={2019}
}
"""


class BNSP(evaluate.EvaluationModule):
    def _info(self):
        return datasets.MetricInfo(
            description="BNSP (Background Negative, Subgroup Positive) AUC",
            citation="https://arxiv.org/abs/1903.04561",
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                 'target': Sequence(feature=Value(dtype='string', id=None),
                                    length=-1, id=None),
                 'label': Value(dtype='int64', id=None),
                 'output': Sequence(feature=Value(dtype='float32', id=None),
                                    length=-1, id=None),
                }
            ),
            reference_urls=["https://rajpurkar.github.io/SQuAD-explorer/"],
        )
           
    def _genreate_subgroup(self,
                           targets,
                           labels,
                           outputs,
                           subgroup,
                           target_class=None):
        """Returns label and output score from `targets` and `labels`
        if `subgroup` is in list of targeted groups found in `targets`
        """
        if target_class is not None:
            target_class = target_class
        else: 
            target_class = np.asarray(outputs).shape[-1] - 1
        for target, label, result in zip(targets, labels, outputs):
            if subgroup in target:
                yield label, result[target_class]
                    
    def _genreate_bpsn(self,
                       targets,
                       labels,
                       outputs,
                       subgroup,
                       target_class=None):
        """Returns label and output score from `targets` and `labels`
        if (1) `subgroup` is in list of targeted groups found in `targets` and
        label is not the same as `target_class`; or (2) `subgroup` is not in
        list of targeted groups found in `targets` and  label is the same as
        `target_class`
        """
        if target_class is not None:
            target_class = target_class
        else: 
            target_class = np.asarray(outputs).shape[-1] - 1
        for target, label, result in zip(targets, labels, outputs):
            if not target:
                continue
            # background positive
            if subgroup not in target and label == target_class:
                yield label, result[target_class]
            # subgroup negative
            elif subgroup in target and label != target_class:
                yield label, result[target_class]
    
    def _genreate_bnsp(self,
                       targets,
                       labels,
                       outputs,
                       subgroup,
                       target_class=None):
        """Returns label and output score from `targets` and `labels`
        if (1) `subgroup` is not in list of targeted groups found in `targets`
        and label is the same as `target_class`; or (2) `subgroup` is in list
        of targeted groups found in `targets` and  label is not the same as
        `target_class`
        """
        # get the index from class
        if target_class is not None:
            target_class = target_class
        else: 
            target_class = np.asarray(outputs).shape[-1] - 1
        for target, label, result in zip(targets, labels, outputs):
            if not target:
                continue
            # background negative
            if subgroup not in target and label != target_class:
                yield label, result[target_class]
            # subgroup positive
            elif subgroup in target and label == target_class:
                yield label, result[target_class]
            
    def _auc_by_group(self, target, label, output, subgroup):
        """ Compute bias AUC metrics 
        """
    
        y_trues, y_preds = zip(*self._genreate_subgroup(target,
                                                        label,
                                                        output,
                                                        subgroup))
        subgroup_auc_score = roc_auc_score(y_trues, y_preds)
        
        y_trues, y_preds = zip(*self._genreate_bpsn(target,
                                                    label,
                                                    output,
                                                    subgroup))
        bpsn_auc_score = roc_auc_score(y_trues, y_preds)

        y_trues, y_preds = zip(*self._genreate_bnsp(target,
                                                    label,
                                                    output,
                                                    subgroup))
        bnsp_auc_score = roc_auc_score(y_trues, y_preds)

        return {'Subgroup': subgroup_auc_score,
                'BPSN': bpsn_auc_score,
                'BNSP': bnsp_auc_score}
    
    def _update_overall(self, result, labels, outputs, power_value=-5):
        """Compute the generalized mean of Bias AUCs"""
        result['Overall'] = {}
        for metric in ['Subgroup', 'BPSN', 'BNSP']:
            metric_values = np.array([result[community][metric]
                                      for community in result
                                      if community != 'Overall'])
            metric_values **= power_value
            mean_value = np.power(np.sum(metric_values)/(len(result) - 1), 
                                  1 / power_value)
            result['Overall'][f"{metric} generalized mean"] = mean_value
        y_preds = [output[1] for output in outputs]
        result['Overall']["Overall AUC"] = roc_auc_score(labels, y_preds)
        return result

    def _compute(self, target, label, output, subgroups=None):
        if subgroups is None:
            subgroups = set(group for group_list in target
                            for group in group_list)
        result = {subgroup: self._auc_by_group(target, label, output, subgroup)
                  for subgroup in subgroups}
        result = self._update_overall(result, label, output)
        return result
