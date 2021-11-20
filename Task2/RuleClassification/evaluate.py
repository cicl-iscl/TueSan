import torch

from tqdm import tqdm
from functools import partial
from sklearn.metrics import accuracy_score
from collections import defaultdict
from stemming_rules import apply_rule
from index_dataset import UNK_TOKEN

from uni2intern import internal_transliteration_to_unicode as to_uni


def get_applicable_rules(token, indexer, tag_rules):
    candidate_stems = []
    candidate_tags = []
    candidate_rule_indices = []

    for rule, index in indexer.stem_rule2index.items():
        try:
            candidate_stem = apply_rule(token, rule)
            candidate_stems.append(candidate_stem)
            candidate_rule_indices.append(index)

            if tag_rules:
                candidate_tags.append(rule[-1])
        except ValueError:
            continue

    return candidate_stems, candidate_tags, candidate_rule_indices


def get_stem_prediction(token, indexer, y_pred_stem, tag_rules):
    candidate_stems, candidate_tags, candidate_rule_indices = get_applicable_rules(
        token, indexer, tag_rules
    )

    if len(candidate_stems) == 0:
        predicted_stem = token

    elif len(candidate_stems) == 1:
        predicted_stem = candidate_stems[0]

        if tag_rules:
            predicted_tag = candidate_tags[0]

    else:
        scores = y_pred_stem[candidate_rule_indices]
        best_index = torch.argmax(scores).cpu().item()
        predicted_stem = candidate_stems[best_index]

        if tag_rules:
            predicted_tag = candidate_tags[best_index]

    if tag_rules:
        return predicted_stem, predicted_tag
    else:
        return predicted_stem


def evaluate_batch(model, batch, indexer, device, tag_rules):
    encoder = model["encoder"]
    stem_rule_classifier = model["stem_rule_classifier"]

    encoder.eval()
    stem_rule_classifier.eval()

    if not tag_rules:
        tag_classifier = model["tag_classifier"]
        tag_classifier.eval()

    raw_tokens, indexed_tokens = batch
    indexed_tokens = indexed_tokens.to(device)

    token_embeddings = encoder(indexed_tokens)
    y_pred_stem = stem_rule_classifier(token_embeddings)

    if not tag_rules:
        y_pred_tag = tag_classifier(token_embeddings)

    token_pointer = 0

    batch_predictions = []
    for sentence in raw_tokens:
        sent_predictions = []

        for token in sentence:
            stem_scores = y_pred_stem[token_pointer]
            if tag_rules:
                predicted_stem, predicted_tag = get_stem_prediction(
                    token, indexer, stem_scores, tag_rules
                )
            else:
                predicted_stem = get_stem_prediction(
                    token, indexer, stem_scores, tag_rules
                )

                tag_scores = y_pred_tag[token_pointer]
                predicted_tag_index = torch.argmax(tag_scores).cpu().item()
                predicted_tag = indexer.index2tag[predicted_tag_index]

            sent_predictions.append([predicted_stem, predicted_tag])
            token_pointer += 1

        batch_predictions.append(sent_predictions)

    assert token_pointer == token_embeddings.shape[0]
    return batch_predictions


def evaluate_model(model, eval_dataloader, indexer, device, tag_rules, translit):
    predictions = []
    with torch.no_grad():
        for batch in eval_dataloader:
            batch_predictions = evaluate_batch(model, batch, indexer, device, tag_rules)
            predictions.extend(batch_predictions)

    if translit:
        predictions = [
            [[to_uni(stem), tag] for stem, tag in sent] for sent in predictions
        ]

    return predictions


def print_metrics(predictions, ground_truth):
    tags_true, tags_pred, stems_true, stems_pred = [], [], [], []
    assert len(predictions) == len(ground_truth)

    for y_true, y_pred in zip(ground_truth, predictions):
        tokens, stems, tags, rules = zip(*y_true)
        assert len(tags) == len(y_pred[0])

        tags_true.extend(tags)
        tags_pred.extend(y_pred[0])

        stems_true.extend(stems)
        stems_pred.extend(y_pred[1])

    tag_accuracy = accuracy_score(tags_pred, tags_true)
    stem_accuracy = accuracy_score(stems_pred, stems_true)

    print()
    print(f"Tag Accuracy: {tag_accuracy:.4f}")
    print(f"Stem Accuracy: {stem_accuracy:.4f}")


def format_predictions(predictions, translit=False):
    preds = []

    tags, stems = zip(*predictions)
    for sent_stems, sent_tags in zip(stems, tags):
        if translit:
            sent_stems = [to_uni(x) for x in sent_stems]
        sentence = list(zip(sent_stems, sent_tags))
        sentence = [list(tup) for tup in sentence]
        preds.append(sentence)
    return preds


def convert_eval_if_translit(eval_data):
    converted = []
    for dp in eval_data:
        converted.append((dp[0], [(to_uni(x), y) for x, y in dp[1]]))
    return converted
