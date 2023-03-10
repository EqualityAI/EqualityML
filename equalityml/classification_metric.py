import numpy as np

import utils
from aif360.datasets import BinaryLabelDataset
from aif360.datasets.multiclass_label_dataset import MulticlassLabelDataset


class ClassificationMetric:
    """Class for computing metrics based on two BinaryLabelDatasets.
    The first dataset is the original one and the second is the output of the
    classification transformer (or similar).
    """

    def __init__(self, dataset, classified_dataset,
                 unprivileged_groups=None, privileged_groups=None):
        """
        Args:
            dataset (BinaryLabelDataset): Dataset containing ground-truth
                labels.
            classified_dataset (BinaryLabelDataset): Dataset containing
                predictions.
            privileged_groups (list(dict)): Privileged groups. Format is a list
                of `dicts` where the keys are `protected_attribute_names` and
                the values are values in `protected_attributes`. Each `dict`
                element describes a single group. See examples for more details.
            unprivileged_groups (list(dict)): Unprivileged groups in the same
                format as `privileged_groups`.
        Raises:
            TypeError: `dataset` and `classified_dataset` must be
                :obj:`~aif360.datasets.BinaryLabelDataset` types.
        """
        if not isinstance(dataset, BinaryLabelDataset) and not isinstance(dataset, MulticlassLabelDataset):
            raise TypeError("'dataset' should be a BinaryLabelDataset or a MulticlassLabelDataset")

        # sets self.dataset, self.unprivileged_groups, self.privileged_groups
        self.dataset = dataset
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups

        if isinstance(classified_dataset, BinaryLabelDataset) or isinstance(classified_dataset,
                                                                            MulticlassLabelDataset):
            self.classified_dataset = classified_dataset
        else:
            raise TypeError("'classified_dataset' should be a "
                            "BinaryLabelDataset or a MulticlassLabelDataset.")

    def _to_condition(self, privileged):
        """Converts a boolean condition to a group-specifying format that can be
        used to create a conditioning vector.
        """
        if privileged is True and self.privileged_groups is None:
            raise AttributeError("'privileged_groups' was not provided when "
                                 "this object was initialized.")
        if privileged is False and self.unprivileged_groups is None:
            raise AttributeError("'unprivileged_groups' was not provided when "
                                 "this object was initialized.")

        if privileged is None:
            return None
        return self.privileged_groups if privileged else self.unprivileged_groups

    def binary_confusion_matrix(self, privileged=None):
        """Compute the number of true/false positives/negatives, optionally
        conditioned on protected attributes.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            dict: Number of true positives, false positives, true negatives,
            false negatives (optionally conditioned).
        """
        condition = self._to_condition(privileged)

        return utils.compute_num_TF_PN(self.dataset.protected_attributes,
                                       self.dataset.labels, self.classified_dataset.labels,
                                       self.dataset.instance_weights,
                                       self.dataset.protected_attribute_names,
                                       self.dataset.favorable_label, self.dataset.unfavorable_label,
                                       condition=condition)

    def generalized_binary_confusion_matrix(self, privileged=None):
        """Compute the number of generalized true/false positives/negatives,
        optionally conditioned on protected attributes. Generalized counts are
        based on scores and not on the hard predictions.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            dict: Number of generalized true positives, generalized false
            positives, generalized true negatives, generalized false negatives
            (optionally conditioned).
        """
        condition = self._to_condition(privileged)

        return utils.compute_num_gen_TF_PN(self.dataset.protected_attributes,
                                           self.dataset.labels, self.classified_dataset.scores,
                                           self.dataset.instance_weights,
                                           self.dataset.protected_attribute_names,
                                           self.dataset.favorable_label, self.dataset.unfavorable_label,
                                           condition=condition)

    def num_true_positives(self, privileged=None):
        r"""Return the number of instances in the dataset where both the
        predicted and true labels are 'favorable',
        :math:`TP = \sum_{i=1}^n \mathbb{1}[y_i = \text{favorable}]\mathbb{1}[\hat{y}_i = \text{favorable}]`,
        optionally conditioned on protected attributes.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['TP']

    def num_false_positives(self, privileged=None):
        r""":math:`FP = \sum_{i=1}^n \mathbb{1}[y_i = \text{unfavorable}]\mathbb{1}[\hat{y}_i = \text{favorable}]`
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['FP']

    def num_false_negatives(self, privileged=None):
        r""":math:`FN = \sum_{i=1}^n \mathbb{1}[y_i = \text{favorable}]\mathbb{1}[\hat{y}_i = \text{unfavorable}]`
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['FN']

    def num_true_negatives(self, privileged=None):
        r""":math:`TN = \sum_{i=1}^n \mathbb{1}[y_i = \text{unfavorable}]\mathbb{1}[\hat{y}_i = \text{unfavorable}]`
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return self.binary_confusion_matrix(privileged=privileged)['TN']

    def num_negatives(self, privileged=None):
        r"""Compute the number of negatives,
        :math:`N = \sum_{i=1}^n \mathbb{1}[y_i = 0]`, optionally conditioned on
        protected attributes.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)
        return utils.compute_num_pos_neg(self.dataset.protected_attributes,
                                         self.dataset.labels, self.dataset.instance_weights,
                                         self.dataset.protected_attribute_names,
                                         self.dataset.unfavorable_label, condition=condition)

    def num_positives(self, privileged=None):
        r"""Compute the number of positives,
        :math:`P = \sum_{i=1}^n \mathbb{1}[y_i = 1]`,
        optionally conditioned on protected attributes.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)
        return utils.compute_num_pos_neg(self.dataset.protected_attributes,
                                         self.dataset.labels, self.dataset.instance_weights,
                                         self.dataset.protected_attribute_names,
                                         self.dataset.favorable_label, condition=condition)

    def performance_measures(self, privileged=None):
        """Compute various performance measures on the dataset, optionally
        conditioned on protected attributes.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Returns:
            dict: True positive rate, true negative rate, false positive rate,
            false negative rate, positive predictive value, negative predictive
            value, false discover rate, false omission rate, and accuracy
            (optionally conditioned).
        """
        TP = self.num_true_positives(privileged=privileged)
        FP = self.num_false_positives(privileged=privileged)
        FN = self.num_false_negatives(privileged=privileged)
        TN = self.num_true_negatives(privileged=privileged)
        GTP = self.generalized_binary_confusion_matrix(privileged=privileged)['GTP']
        GFP = self.generalized_binary_confusion_matrix(privileged=privileged)['GFP']
        GFN = self.generalized_binary_confusion_matrix(privileged=privileged)['GFN']
        GTN = self.generalized_binary_confusion_matrix(privileged=privileged)['GTN']
        P = self.num_positives(privileged=privileged)
        N = self.num_negatives(privileged=privileged)

        return dict(
            TPR=TP / P, TNR=TN / N, FPR=FP / N, FNR=FN / P,
            GTPR=GTP / P, GTNR=GTN / N, GFPR=GFP / N, GFNR=GFN / P,
            PPV=TP / (TP + FP) if (TP + FP) > 0.0 else np.float64(0.0),
            NPV=TN / (TN + FN) if (TN + FN) > 0.0 else np.float64(0.0),
            FDR=FP / (FP + TP) if (FP + TP) > 0.0 else np.float64(0.0),
            FOR=FN / (FN + TN) if (FN + TN) > 0.0 else np.float64(0.0),
            ACC=(TP + TN) / (P + N) if (P + N) > 0.0 else np.float64(0.0)
        )


    def num_instances(self, privileged=None):
        """Compute the number of instances, :math:`n`, in the dataset conditioned
        on protected attributes if necessary.
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups` must be
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)
        return utils.compute_num_instances(self.dataset.protected_attributes,
            self.dataset.instance_weights,
            self.dataset.protected_attribute_names, condition=condition)

    def num_pred_positives(self, privileged=None):
        r""":math:`\sum_{i=1}^n \mathbb{1}[\hat{y}_i = \text{favorable}]`
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        condition = self._to_condition(privileged)

        return utils.compute_num_pos_neg(
            self.classified_dataset.protected_attributes,
            self.classified_dataset.labels,
            self.classified_dataset.instance_weights,
            self.classified_dataset.protected_attribute_names,
            self.classified_dataset.favorable_label,
            condition=condition)

    def selection_rate(self, privileged=None):
        r""":math:`Pr(\hat{Y} = \text{favorable})`
        Args:
            privileged (bool, optional): Boolean prescribing whether to
                condition this metric on the `privileged_groups`, if `True`, or
                the `unprivileged_groups`, if `False`. Defaults to `None`
                meaning this metric is computed over the entire dataset.
        Raises:
            AttributeError: `privileged_groups` or `unprivileged_groups`
                must be provided at initialization to condition on them.
        """
        return (self.num_pred_positives(privileged=privileged)
              / self.num_instances(privileged=privileged))
