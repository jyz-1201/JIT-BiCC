from __future__ import absolute_import, division, print_function

import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader, SequentialSampler


sys.path.append("/mnt/sda/jyz/Code_Change/CommitMaster/")  # Change to root directory of your project

from JITFine.my_util import TextDataset, get_line_level_metrics
from JITFine.concat.run import commit_with_codes

logger = logging.getLogger(__name__)


def test_interpret(args, model, tokenizer, best_threshold=0.5):
    # build dataloader
    cache_dataset = os.path.dirname(args.test_data_file[0]) + f'/test_set_cache_msg{args.max_msg_length}.pkl'
    if args.no_abstraction:
        cache_dataset = cache_dataset.split('.pkl')[0] + '_raw.pkl'
    logger.info("Cache Dataset file at %s ", cache_dataset)
    if os.path.exists(cache_dataset):
        eval_dataset = pickle.load(open(cache_dataset, 'rb'))
    else:
        eval_dataset = TextDataset(tokenizer, args, file_path=args.test_data_file, mode='test')
        pickle.dump(eval_dataset, open(cache_dataset, 'wb'))
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits = []
    y_trues = []

    integrated_gradients = IntegratedGradients(model)
    # todo: initialize a python list

    for batch in eval_dataloader:
        (inputs_ids, attn_masks, manual_features, labels) = [x.to(args.device) for x in batch]
        with torch.no_grad():
            loss, logit = model(inputs_ids, attn_masks, manual_features, labels)
            eval_loss += loss.mean().item()
            logits.append(logit.cpu().numpy())
            y_trues.append(labels.cpu().numpy())

            attributions_ig = integrated_gradients.attribute(inputs_ids, target=logit, n_steps=200)
            # todo: append attributions_ig into a python list

        nb_eval_steps += 1
    # output result
    # calculate scores
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)

    y_preds = logits[:, -1] > best_threshold
    from sklearn.metrics import recall_score
    recall = recall_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import precision_score
    precision = precision_score(y_trues, y_preds, average='binary')
    from sklearn.metrics import f1_score
    f1 = f1_score(y_trues, y_preds, average='binary')

    result = {
        "eval_recall": float(recall),
        "eval_precision": float(precision),
        "eval_f1": float(f1),
        "eval_threshold": best_threshold,

    }
    logger.info("***** Eval results *****")
    for key in sorted(result.keys()):
        logger.info("  %s = %s", key, str(round(result[key], 4)))

    result = []

    cache_buggy_line = os.path.join(os.path.dirname(args.buggy_line_filepath),
                                    'changes_complete_buggy_line_level_cache.pkl')
    if os.path.exists(cache_buggy_line):
        commit2codes, idx2label = pickle.load(open(cache_buggy_line, 'rb'))
    else:
        commit2codes, idx2label = commit_with_codes(args.buggy_line_filepath, tokenizer)
        pickle.dump((commit2codes, idx2label), open(cache_buggy_line, 'wb'))

    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = [], [], [], [], []
    for example, pred, prob in zip(eval_dataset.examples, y_preds, logits[:, -1]):
        result.append([example.commit_id, prob, pred, example.label])

        # calculate
        if int(example.label) == 1 and int(pred) == 1 and '[ADD]' in example.input_tokens:
            cur_codes = commit2codes[commit2codes['commit_id'] == example.commit_id]
            cur_labels = idx2label[idx2label['commit_id'] == example.commit_id]
            # todo: pass the python list into deal_with_attributions
            cur_IFA, cur_top_20_percent_LOC_recall, cur_effort_at_20_percent_LOC_recall, cur_top_10_acc, \
                cur_top_5_acc = deal_with_attributions(example, pred, cur_codes, cur_labels, args.only_adds)
            IFA.append(cur_IFA)
            top_20_percent_LOC_recall.append(cur_top_20_percent_LOC_recall)
            effort_at_20_percent_LOC_recall.append(cur_effort_at_20_percent_LOC_recall)
            top_10_acc.append(cur_top_10_acc)
            top_5_acc.append(cur_top_5_acc)

    logger.info(
        'Top-10-ACC: {:.4f},Top-5-ACC: {:.4f}, Recall20%Effort: {:.4f}, Effort@20%LOC: {:.4f}, IFA: {:.4f}'.format(
            round(np.mean(top_10_acc), 4), round(np.mean(top_5_acc), 4),
            round(np.mean(top_20_percent_LOC_recall), 4),
            round(np.mean(effort_at_20_percent_LOC_recall), 4), round(np.mean(IFA), 4))
    )
    RF_result = pd.DataFrame(result)
    RF_result.to_csv(os.path.join(args.output_dir, "predictions.csv"), sep='\t', index=None)


def deal_with_attributions(item, pred, commit2codes, idx2label, only_adds=False):
    # todo: pass the python list into deal_with_attributions and use it instead of attns
    commit_id = item.commit_id
    input_tokens = item.input_tokens
    commit_label = item.label

    # remove msg,cls,eos,del
    begin_pos = input_tokens.index('[ADD]')
    end_pos = input_tokens.index('[DEL]') if '[DEL]' in input_tokens else len(input_tokens) - 1

    attn_df = pd.DataFrame()
    attn_df['token'] = [token.replace('\u0120', '') for token in
                        input_tokens[begin_pos:end_pos]]
    # average score for multi-heads
    attns = attns.mean(axis=0)[begin_pos:end_pos]
    attn_df['score'] = attns
    attn_df = attn_df.sort_values(by='score', ascending=False)
    attn_df = attn_df.groupby('token').sum()
    attn_df['token'] = attn_df.index
    attn_df = attn_df.reset_index(drop=True)

    # calculate score for each line in commit
    if only_adds:
        commit2codes = commit2codes[commit2codes['changed_type'] == 'added']  # only count for added lines
    commit2codes = commit2codes.drop('commit_id', axis=1)
    commit2codes = commit2codes.drop('changed_type', axis=1)

    result_df = pd.merge(commit2codes, attn_df, how='left', on='token')
    result_df = result_df.groupby(['idx']).sum()
    result_df = result_df.reset_index(drop=False)

    result_df = pd.merge(result_df, idx2label, how='inner', on='idx')
    IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc = get_line_level_metrics(
        result_df['score'].tolist(), result_df['label'].tolist())
    return IFA, top_20_percent_LOC_recall, effort_at_20_percent_LOC_recall, top_10_acc, top_5_acc
