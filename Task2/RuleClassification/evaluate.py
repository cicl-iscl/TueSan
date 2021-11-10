import torch

from tqdm import tqdm
from functools import partial
from sklearn.metrics import accuracy_score
from training import get_loss
from extract_rules import rule_is_applicable
from extract_rules import UNK_RULE
from vocabulary import UNK_TOKEN
from collections import defaultdict

from uni2intern import internal_transliteration_to_unicode as to_uni


def evaluate_model(model, eval_dataloader, device, rules, rule_encoder, tag_encoder):
    index2rule = defaultdict(lambda: UNK_RULE)
    index2rule.update({index: rule for rule, index in rule_encoder.items()})

    if tag_encoder is not None:
        index2tag = defaultdict(lambda: UNK_TOKEN)
        index2tag.update({index: tag for tag, index in tag_encoder.items()})

    predictions = []

    encoder = model["encoder"]
    classifiers = model["classifiers"]

    encoder.eval()
    classifiers.eval()

    batch_processor = partial(
        get_loss, encoder=encoder, classifiers=classifiers, device=device
    )

    with torch.no_grad():
        for batch in tqdm(eval_dataloader):
            raw_tokens, raw_stems, *batch = batch
            loss, _, (y_pred_rule, y_pred_tag) = batch_processor(batch)

            running_index = 0

            for sentence, stems in zip(raw_tokens, raw_stems):
                tag_predictions = []
                stem_predictions = []

                for token, stem in zip(sentence, stems):
                    rule_probs = y_pred_rule[running_index]
                    candidate_stems = []
                    candidate_rules = []

                    # Find applicable rules
                    for rule in rules:
                        is_applicable, candidate_stem = rule_is_applicable(rule, token)
                        if is_applicable:
                            candidate_rules.append(rule)
                            candidate_stems.append(candidate_stem)

                    indexed_candidate_rules = [
                        rule_encoder[rule] for rule in candidate_rules
                    ]
                    indexed_candidate_rules = torch.LongTensor(indexed_candidate_rules)

                    if len(candidate_rules) == 0:
                        predicted_rule = UNK_RULE
                        predicted_stem = token
                        predicted_tag = UNK_TOKEN

                    elif len(candidate_rules) == 1:
                        predicted_rule = candidate_rules[0]
                        predicted_stem = candidate_stems[0]
                        if len(predicted_rule) == 4:
                            predicted_tag = predicted_rule[3]
                        else:
                            predicted_tag = None

                    else:
                        candidate_rule_probs = rule_probs[indexed_candidate_rules]
                        best_rule_index = torch.argmax(candidate_rule_probs).item()

                        predicted_rule = candidate_rules[best_rule_index]
                        predicted_stem = candidate_stems[best_rule_index]

                        if len(predicted_rule) == 4:
                            predicted_tag = predicted_rule[3]
                        else:
                            predicted_tag = None

                    if y_pred_tag is not None:
                        tag_probs = y_pred_tag[running_index]
                        best_tag_index = torch.argmax(tag_probs).item()
                        predicted_tag = index2tag[best_tag_index]

                    # sentence_predictions.append((token, stem, predicted_rule, predicted_stem, predicted_tag))
                    stem_predictions.append(predicted_stem)
                    tag_predictions.append(predicted_tag)
                    running_index += 1

                predictions.append((tag_predictions, stem_predictions))
            assert running_index == y_pred_rule.shape[0]
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
        preds.append(sentence)
    return preds
