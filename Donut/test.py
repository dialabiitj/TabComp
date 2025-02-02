import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance
from anls import anls_score

from donut import DonutModel, JSONParseEvaluator, load_json, save_json
import nltk
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

import bert_score

nltk.download('punkt')
nltk.download('wordnet')
def calculate_meteor_score(prediction, references):
    filtered_ground_truths = [gt['answer'] for gt in references if gt['question'].lower() == prediction['question'].lower()]

    # Tokenize the prediction answer
    prediction_answer = prediction.get('answer')

    if prediction_answer:
        tokenized_prediction = nltk.word_tokenize(prediction_answer)
        # Calculate METEOR score
        if filtered_ground_truths:
            meteor_scores = []
            for gt in filtered_ground_truths:
                if gt:
                    tokenized_gt = nltk.word_tokenize(gt)
                    score = meteor_score([tokenized_gt], tokenized_prediction)
                    meteor_scores.append(score)
            average_meteor_score = sum(meteor_scores) / len(meteor_scores)
            return average_meteor_score
        else:
            return 0
    else:
        return 0

def calculate_bleu_scores(prediction, references):
    bleu_scores = {
        "bleu_1": 0,
        "bleu_2": 0,
        "bleu_3": 0,
        "bleu_4": 0
    }
    if prediction:
        tokenized_prediction = nltk.word_tokenize(prediction)
        tokenized_references = [nltk.word_tokenize(ref) for ref in references]
        smoothie = SmoothingFunction().method4

        bleu_scores["bleu_1"] = sentence_bleu(tokenized_references, tokenized_prediction, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_scores["bleu_2"] = sentence_bleu(tokenized_references, tokenized_prediction, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_scores["bleu_3"] = sentence_bleu(tokenized_references, tokenized_prediction, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu_scores["bleu_4"] = sentence_bleu(tokenized_references, tokenized_prediction, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    return bleu_scores

def calculate_rouge_score(prediction, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    if prediction:
        for ref in references:
            score = scorer.score(ref, prediction)
            scores["rouge1"] += score['rouge1'].fmeasure
            scores["rouge2"] += score['rouge2'].fmeasure
            scores["rougeL"] += score['rougeL'].fmeasure

        scores = {k: v / len(references) for k, v in scores.items()}
    return scores

def calculate_bert_score(prediction, references):
    P, R, F1 = bert_score.score([prediction] * len(references), references, lang="en", verbose=False)
    return F1.mean().item()


def test(args):
    try:
        pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)
    except Exception as e:
        raise ValueError(f"Error loading pretrained model: {e}")

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []
    ground_truths = []
    accs = []
    anls = []
    meteor = []
    bleu_scores_list = []
    rouge_scores_list = []
    bert_f1_scores = []

    evaluator = JSONParseEvaluator()

    try:
        dataset = load_dataset(args.dataset_name_or_path, split=args.split)
    except Exception as e:
        raise ValueError(f"Error loading dataset: {e}")

    for idx, sample in tqdm(enumerate(dataset), total=len(dataset)):
        ground_truth = json.loads(sample["ground_truth"])

        output = pretrained_model.inference(
            image=sample["image"],
            prompt=f"<s_{args.task_name}><s_question>{ground_truth['gt_parses'][0]['question'].lower()}</s_question><s_answer>",
        )["predictions"][0]
        
        gt = ground_truth["gt_parses"]
        answers = set([qa_parse["answer"] for qa_parse in gt])
        
        try:
            score = float(output["answer"] in answers)
            p = output['answer']
        except KeyError:
            score = 0.0
            p = ""
        
        g_truth = list(answers)
        
        accs.append(score)
        predictions.append(output)
        ground_truths.append(gt)
        anls.append(anls_score(prediction=p, gold_labels=g_truth, threshold=0.5))
        meteor.append(calculate_meteor_score(output, gt))

        # Calculate BLEU scores
        bleu_scores = calculate_bleu_scores(p, g_truth)
        bleu_scores_list.append(bleu_scores)

        # Calculate ROUGE scores
        rouge_scores = calculate_rouge_score(p, g_truth)
        rouge_scores_list.append(rouge_scores)

        # Calculate BERTScore F1
        if p:
            bert_f1 = calculate_bert_score(p, g_truth)
            bert_f1_scores.append(bert_f1)
        else:
            bert_f1_scores.append(0)

    avg_bleu_scores = {k: np.mean([d[k] for d in bleu_scores_list]) for k in bleu_scores_list[0]}
    avg_rouge_scores = {k: np.mean([d[k] for d in rouge_scores_list]) for k in rouge_scores_list[0]}
    avg_bert_f1 = np.mean(bert_f1_scores)

    scores = {
        "ted_accuracies": accs,
        "ted_accuracy": np.mean(accs),
        "f1_accuracy": evaluator.cal_f1(predictions, ground_truths),
        "anls_score": np.mean(anls),
        "meteor_score": np.mean(meteor),
        "bleu_scores": avg_bleu_scores,
        "rouge_scores": avg_rouge_scores,
        "bert_f1_score": avg_bert_f1
    }

    print(
        f"Total number of samples: {len(accs)}, Tree Edit Distance (TED) based accuracy score: {scores['ted_accuracy']}, "
        f"F1 accuracy score: {scores['f1_accuracy']}, ANLS score: {scores['anls_score']}, METEOR score: {scores['meteor_score']}, "
        f"BLEU scores: {scores['bleu_scores']}, ROUGE scores: {scores['rouge_scores']}, BERTScore F1: {scores['bert_f1_score']}"
    )

    if args.save_path:
        scores["predictions"] = predictions
        scores["ground_truths"] = ground_truths
        save_json(args.save_path, scores)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True, help="Path to the pretrained model or Hugging Face model identifier")
    parser.add_argument("--dataset_name_or_path", type=str, required=True, help="Path to the dataset or Hugging Face dataset identifier")
    parser.add_argument("--split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--task_name", type=str, default=None, help="Name of the task")
    parser.add_argument("--save_path", type=str, default=None, help="Path to save the results")
    args, left_argv = parser.parse_known_args()

    if args.task_name is None:
        args.task_name = os.path.basename(args.dataset_name_or_path)

    predictions = test(args)

