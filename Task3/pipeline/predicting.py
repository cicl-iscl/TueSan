import torch

from tqdm import tqdm
from reconstruct_translit import reconstruct_unsandhied as reconstruct_translit
from reconstruct_uni import reconstruct_unsandhied
from uni2intern import internal_transliteration_to_unicode as to_uni

from sandhi_rules import KEEP_RULE, APPEND_SPACE_RULE
from generate_dataset import get_boundaries
from stemming_rules import apply_rule


def unsandhi(chars, rules):
    current_prediction = []
    previous_rule = None

    for char, rule in zip(chars, rules):
        if rule == KEEP_RULE or char == " ":
            current_prediction.append(char)
        elif rule == APPEND_SPACE_RULE:
            current_prediction.append(char)
            current_prediction.append(" ")

        elif rule != previous_rule:
            current_prediction.append(rule)
        else:
            pass

        previous_rule = rule

    return "".join(current_prediction)


def get_valid_rules(token, indexer):
    candidate_stems = []
    candidate_stem_indices = []
    for rule in indexer.stem_rules:
        try:
            candidate_stem = apply_rule(token, rule)
            candidate_stem_index = indexer.stem_rule2index[rule]
            candidate_stems.append(candidate_stem)
            candidate_stem_indices.append(candidate_stem_index)
        except ValueError:
            continue

    return candidate_stems, candidate_stem_indices


def make_predictions(model, eval_dataloader, indexer, device, translit=False):
    model = model.to(device)

    segmenter = model["segmenter"]
    classifier = model["classifier"]

    segmenter.eval()
    classifier.eval()

    predictions = []

    with torch.no_grad():
        for raw_source, source in tqdm(eval_dataloader):
            source = source.to(device)
            y_pred_sandhi, char_embeddings = segmenter(source)
            predicted_rules = torch.argmax(y_pred_sandhi, dim=-1)
            predicted_rules = predicted_rules.cpu().long().numpy().tolist()

            lengths = (source != 0).sum(dim=-1).cpu().flatten().long().numpy().tolist()
            source = source.cpu().long().numpy().tolist()

            raw_rules = []
            boundaries = []

            batch_sentences = []

            for raw_chars, char_indices, rule_indices, length in zip(
                raw_source, source, predicted_rules, lengths
            ):
                char_indices = char_indices[:length]
                rule_indices = rule_indices[:length]

                chars = raw_chars
                rules = indexer.restore_indexed_sandhi_rules(rule_indices)

                raw_rules.append(rules)
                current_boundaries = get_boundaries(chars, rules)
                boundaries.append(current_boundaries)

                current_prediction = unsandhi(chars, rules)

                sent_tokens = current_prediction.split(" ")
                try:
                    assert len(current_boundaries) == len(sent_tokens)
                except:
                    print(current_boundaries)
                    print(sent_tokens)
                    print(rules)
                    print(chars)
                    print()
                    raise

                batch_sentences.append(sent_tokens)

            y_pred_stem, y_pred_tag = classifier(char_embeddings, boundaries)
            stem_predictions = (
                torch.argmax(y_pred_stem, dim=-1).cpu().flatten().numpy().tolist()
            )
            stem_predictions = indexer.restore_indexed_stem_rules(stem_predictions)
            tag_predictions = (
                torch.argmax(y_pred_stem, dim=-1).cpu().flatten().numpy().tolist()
            )
            tag_predictions = indexer.restore_indexed_tags(tag_predictions)

            # batch_num_tokens = [len(tokens) for tokens in boundaries]
            token_pointer = 0
            for sentence in batch_sentences:
                sentence_prediction = []

                for token in sentence:
                    candidate_stems, candidate_stem_indices = get_valid_rules(
                        token, indexer
                    )
                    stem_scores = y_pred_stem[token_pointer]

                    if len(candidate_stems) == 0:
                        current_stem = token
                    elif len(candidate_stems) == 1:
                        current_stem = candidate_stems[0]
                    else:
                        candidate_scores = stem_scores[candidate_stem_indices]
                        best_index = torch.argmax(candidate_scores).cpu().item()
                        current_stem = candidate_stems[best_index]

                    tag_scores = y_pred_tag[token_pointer]
                    tag_index = torch.argmax(tag_scores).cpu().item()
                    current_tag = indexer.index2tag[tag_index]

                    token_pointer += 1
                    if translit:
                        token = to_uni(token)
                        current_stem = to_uni(current_stem)

                    sentence_prediction.append([token, current_stem, current_tag])

                predictions.append(sentence_prediction)

        return predictions
