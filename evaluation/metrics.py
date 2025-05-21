import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../bart_score')))
from bart_score import BARTScorer

import evaluate
import numpy as np
from bert_score import BERTScorer
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="amazon", help="amazon, yelp or google")
args = parser.parse_args()

class MetricScore:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        self.pred_input_path = (f"data/{args.dataset}/tst_pred.pkl")
        self.ref_input_path = f"data/{args.dataset}/tst_ref.pkl"

        with open(self.pred_input_path, "rb") as f:
            self.data = pickle.load(f)
        with open(self.ref_input_path, "rb") as f:
            self.ref_data = pickle.load(f)

    def get_score(self):
        scores = {}
        (
            bert_precison,
            bert_recall,
            bert_f1,
            bert_precison_std,
            bert_recall_std,
            bert_f1_std,
        ) = BERT_score(self.data, self.ref_data)
        tokens_predict = [s.split() for s in self.data]
        usr, _ = unique_sentence_percent(tokens_predict)

        # BLEURT
        bleurt_score, bleurt_std = BLEURT_score(self.data, self.ref_data)
        # BARTScore
        bart_score, bart_std = BART_score(self.data, self.ref_data)

        scores["bert_precision"] = bert_precison
        scores["bert_recall"] = bert_recall
        scores["bert_f1"] = bert_f1
        scores["usr"] = usr
        scores["bert_precision_std"] = bert_precison_std
        scores["bert_recall_std"] = bert_recall_std
        scores["bert_f1_std"] = bert_f1_std
        scores["bleurt"] = bleurt_score
        scores["bleurt_std"] = bleurt_std
        scores["bart_score"] = bart_score
        scores["bart_std"] = bart_std
        return scores

    def print_score(self):
        scores = self.get_score()
        print(f"dataset: {args.dataset}")
        print("Explanability Evaluation Metrics:")
        print(f"bert_precision: {scores['bert_precision']:.4f}")
        print(f"bert_recall: {scores['bert_recall']:.4f}")
        print(f"bert_f1: {scores['bert_f1']:.4f}")
        print(f"bleurt: {scores['bleurt']:.4f}")
        print(f"bart_score: {scores['bart_score']:.4f}")
        print(f"usr: {scores['usr']:.4f}")
        print("-"*30)
        print("Standard Deviation:")
        print(f"bert_precision_std: {scores['bert_precision_std']:.4f}")
        print(f"bert_recall_std: {scores['bert_recall_std']:.4f}")
        print(f"bert_f1_std: {scores['bert_f1_std']:.4f}")
        print(f"bleurt_std: {scores['bleurt_std']:.4f}")
        print(f"bart_std: {scores['bart_std']:.4f}")

def two_seq_same(sa, sb):
    if len(sa) != len(sb):
        return False
    for wa, wb in zip(sa, sb):
        if wa != wb:
            return False
    return True

def unique_sentence_percent(sequence_batch):
    unique_seq = []
    for seq in sequence_batch:
        # seq is a list of words
        count = 0
        for uni_seq in unique_seq:
            if two_seq_same(seq, uni_seq):
                count += 1
                break
        if count == 0:
            unique_seq.append(seq)

    return len(unique_seq) / len(sequence_batch), len(unique_seq)

def BERT_score(predictions, references):
    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
    P, R, F1 = scorer.score(predictions, references)
    precision = P.tolist()
    recall = R.tolist()
    f1 = F1.tolist()
    return (
        np.mean(precision),
        np.mean(recall),
        np.mean(f1),
        np.std(precision),
        np.std(recall),
        np.std(f1),
    )

def BLEURT_score(predictions, references):
    bleurt = evaluate.load("bleurt", "bleurt-base-128")
    results = bleurt.compute(predictions=predictions, references=references)
    scores = results["scores"]
    return np.mean(scores), np.std(scores)

def BART_score(predictions, references):
    scorer = BARTScorer(device="cpu", checkpoint="facebook/bart-large-cnn")
    scores = scorer.score(predictions, references)
    scores = [float(s) for s in scores]
    return np.mean(scores), np.std(scores)
