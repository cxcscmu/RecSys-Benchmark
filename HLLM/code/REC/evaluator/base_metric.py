# Copyright (c) 2024 westlake-repl
# SPDX-License-Identifier: MIT

import torch
import logging
import traceback
from REC.utils import EvaluatorType
import numpy as np

logger = logging.getLogger(__name__)


def _slice_rows_by_num_users(dataobject, arr):
    """Return arr[:n_users] if 'data.num_users' is present; else return arr."""
    try:
        if 'data.num_users' in dataobject:
            num = dataobject.get('data.num_users')
            n = int(num.item()) if isinstance(num, torch.Tensor) else int(num)
            if n >= 0:
                return arr[:n]
    except Exception as exc:
        try:
            user_count = dataobject.get('data.num_users') if 'data.num_users' in dataobject else None
            logger.exception(
                "_slice_rows_by_num_users failed (data.num_users=%r, arr_type=%s): %s",
                user_count, type(arr), exc,
            )
        except Exception:
            traceback.print_exc()
    return arr


def _mask_padded_items(dataobject, pos_index_bool):
    """Apply a boolean mask that zeros-out positions where item ids are padded/out-of-range.

    Expects:
      - dataobject['rec.items']: item-id tensor/ndarray of shape [U, K_or_more]
      - dataobject['data.num_items']: total number of valid items (int or tensor scalar)

    Returns:
      pos_index_bool masked in-place semantics (returns a new ndarray).
    """
    try:
        if 'data.num_items' not in dataobject or 'rec.items' not in dataobject:
            return pos_index_bool

        num_items_val = dataobject.get('data.num_items')
        n_items = int(num_items_val.item()) if isinstance(num_items_val, torch.Tensor) else int(num_items_val)
        if n_items < 0:
            return pos_index_bool

        items_t = dataobject.get('rec.items')
        # robustly convert to numpy
        if isinstance(items_t, torch.Tensor):
            items_np = items_t.detach().cpu().numpy()
        else:
            items_np = np.asarray(items_t)

        k = pos_index_bool.shape[1]
        items_np = items_np[:, :k]  # align with K used in pos_index_bool

        valid_mask = (items_np != 0) & (items_np < n_items)
        return pos_index_bool & valid_mask
    except Exception:
        # Fail-safe: if anything goes wrong, return original boolean matrix
        return pos_index_bool


class AbstractMetric(object):
    """:class:`AbstractMetric` is the base object of all metrics. If you want to
        implement a metric, you should inherit this class.

    Args:
        config (Config): the config of evaluator.
    """
    smaller = False

    def __init__(self, config):
        self.decimal_place = config['metric_decimal_place'] + 2 if config['metric_decimal_place'] else 7

    def calculate_metric(self, dataobject):
        """Get the dictionary of a metric.

        Args:
            dataobject(DataStruct): it contains all the information needed to calculate metrics.

        Returns:
            dict: such as ``{'metric@10': 3153, 'metric@20': 0.3824}``
        """
        raise NotImplementedError('Method [calculate_metric] should be implemented.')


class TopkMetric(AbstractMetric):
    """:class:`TopkMetric` is a base object of top-k metrics. If you want to
    implement an top-k metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """
    metric_type = EvaluatorType.RANKING
    metric_need = ['rec.topk']

    def __init__(self, config):
        super().__init__(config)
        self.topk = config['topk']

    def used_info(self, dataobject):
        """Return (pos_index_bool[K], pos_len) after trimming padded users and masking padded items."""
        rec_mat = _slice_rows_by_num_users(dataobject, dataobject.get('rec.topk'))

        kmax = max(self.topk)
        topk_idx, pos_len_list = torch.split(rec_mat, [kmax, 1], dim=1)

        # to numpy for metric math
        pos_index_bool = topk_idx.to(torch.bool).cpu().numpy()
        pos_len_np = pos_len_list.squeeze(-1).cpu().numpy()

        # mask out padded/out-of-range items
        pos_index_bool = _mask_padded_items(dataobject, pos_index_bool)

        return pos_index_bool, pos_len_np

    def topk_result(self, metric, value):
        """Match the metric value to the `k` and put them in `dictionary` form.

        Args:
            metric(str): the name of calculated metric.
            value(numpy.ndarray): metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.

        Returns:
            dict: metric values required in the configuration.
        """
        metric_dict = {}
        avg_result = value.sum(axis=0)
        for k in self.topk:
            key = '{}@{}'.format(metric, k)
            # metric_dict[key] = round(avg_result[k - 1], self.decimal_place)
            metric_dict[key] = avg_result[k - 1]
        return metric_dict

    def metric_info(self, pos_index, pos_len=None):
        """Calculate the value of the metric.

        Args:
            pos_index(numpy.ndarray): a bool matrix, shape of ``n_users * max(topk)``. The item with the (j+1)-th \
            highest score of i-th user is positive if ``pos_index[i][j] == True`` and negative otherwise.
            pos_len(numpy.ndarray): a vector representing the number of positive items per user, shape of ``(n_users,)``.

        Returns:
            numpy.ndarray: metrics for each user, including values from `metric@1` to `metric@max(self.topk)`.
        """
        raise NotImplementedError('Method [metric_info] of top-k metric should be implemented.')


class LossMetric(AbstractMetric):
    """:class:`LossMetric` is a base object of loss based metrics and AUC. If you want to
    implement an loss based metric, you can inherit this class.

    Args:
        config (Config): The config of evaluator.
    """
    metric_type = EvaluatorType.VALUE
    metric_need = ['rec.score', 'data.label']

    def __init__(self, config):
        super().__init__(config)

    def used_info(self, dataobject):
        """Get scores that model predicted and the ground truth."""
        preds = dataobject.get('rec.score')
        trues = dataobject.get('data.label')

        return preds.squeeze(-1).numpy(), trues.squeeze(-1).numpy()

    def output_metric(self, metric, dataobject):
        preds, trues = self.used_info(dataobject)
        result = self.metric_info(preds, trues)
        return {metric: round(result, self.decimal_place)}

    def metric_info(self, preds, trues):
        """Calculate the value of the metric.

        Args:
            preds (numpy.ndarray): the scores predicted by model, a one-dimensional vector.
            trues (numpy.ndarray): the label of items, which has the same shape as ``preds``.

        Returns:
            float: The value of the metric.
        """
        raise NotImplementedError('Method [metric_info] of loss-based metric should be implemented.')
