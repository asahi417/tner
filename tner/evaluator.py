import logging
import os
import json
from seqeval.metrics import f1_score, precision_score, recall_score, classification_report


def evaluate(model,
             export_dir,
             batch_size,
             max_length,
             data):
    path_metric = '{}/metric.json'.format(export_dir)
    if os.path.exists(path_metric):
        with open(path_metric, 'r') as f:
            metric = json.load(f)
        return metric
    os.makedirs(export_dir, exist_ok=True)
    if model is not None:
        lm = TransformersNER(model)
        lm.eval()

    seq_pred, seq_true = [], []
    for encode in data_loader:
        encode = {k: v.to(self.device) for k, v in encode.items()}
        labels_tensor = encode.pop('labels')
        logit = self.model(**encode, return_dict=True)['logits']
        _true = labels_tensor.cpu().detach().int().tolist()
        _pred = torch.max(logit, 2)[1].cpu().detach().int().tolist()
        for b in range(len(_true)):
            _pred_list, _true_list = [], []
            for s in range(len(_true[b])):
                if _true[b][s] != PAD_TOKEN_LABEL_ID:
                    _true_list.append(self.id_to_label[_true[b][s]])
                    if unseen_entity_set is None:
                        _pred_list.append(self.id_to_label[_pred[b][s]])
                    else:
                        __pred = self.id_to_label[_pred[b][s]]
                        if __pred in unseen_entity_set:
                            _pred_list.append('O')
                        else:
                            _pred_list.append(__pred)
            assert len(_pred_list) == len(_true_list)
            if len(_true_list) > 0:
                if entity_span_prediction:
                    # ignore entity type and focus on entity position
                    _true_list = [i if i == 'O' else '-'.join([i.split('-')[0], 'entity']) for i in _true_list]
                    _pred_list = [i if i == 'O' else '-'.join([i.split('-')[0], 'entity']) for i in _pred_list]
                seq_true.append(_true_list)
                seq_pred.append(_pred_list)

    # compute metrics
    metric = {
        "f1": f1_score(seq_true, seq_pred) * 100,
        "recall": recall_score(seq_true, seq_pred) * 100,
        "precision": precision_score(seq_true, seq_pred) * 100,
    }

    try:
        summary = classification_report(seq_true, seq_pred)
        logging.info('[epoch {}] ({}) \n {}'.format(self.__epoch, prefix, summary))
    except Exception:
        logging.exception('classification_report raises error')
        summary = ''
    metric['summary'] = summary
    return metric