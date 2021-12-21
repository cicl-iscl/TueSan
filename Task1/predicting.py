import torch

from tqdm import tqdm
from reconstruct_translit import reconstruct_unsandhied as reconstruct_translit
from reconstruct_uni import reconstruct_unsandhied
from uni2intern import internal_transliteration_to_unicode as to_uni
from uni2intern import unicode_to_internal_transliteration as to_intern

from generate_dataset import KEEP_RULE, APPEND_SPACE_RULE


def make_predictions(model, eval_dataloader, indexer, device, translit=False):
    model = model.to(device)
    model.eval()

    predictions = []

    with torch.no_grad():
        for source in tqdm(eval_dataloader):
            predicted_rules = model(source.to(device))
            predicted_rules = torch.argmax(predicted_rules, dim=-1)
            predicted_rules = predicted_rules.cpu().long().numpy().tolist()

            lengths = (source != 0).sum(dim=-1).flatten().long().numpy().tolist()
            source = source.cpu().long().numpy().tolist()

            for char_indices, rule_indices, length in zip(
                source, predicted_rules, lengths
            ):
                char_indices = char_indices[:length]
                rule_indices = rule_indices[:length]

                chars = indexer.restore_indexed_sent(char_indices)
                rules = indexer.restore_indexed_rules(rule_indices)

                current_prediction = []
                previous_rule = None

                for char, rule in zip(chars, rules):
                    if rule == KEEP_RULE:
                        current_prediction.append(char)
                    elif rule == APPEND_SPACE_RULE:
                        current_prediction.append(char)
                        current_prediction.append(" ")

                    elif rule != previous_rule:
                        current_prediction.append(rule)
                    else:
                        pass

                    previous_rule = rule

                predictions.append("".join(current_prediction))

        if translit:
            predictions = list(map(to_uni, predictions))

        return predictions


def wrong_predictions(
    model,
    eval_dataloader,
    indexer,
    device,
    true_unsandhied,
    translit=False,
    batch_size=64,
):
    model = model.to(device)
    model.eval()

    predictions = []
    wrong_predictions = []
    rules_used = []

    with torch.no_grad():
        for i, source in tqdm(enumerate(eval_dataloader)):
            predicted_sent = ""
            gold = []
            # --------------------------
            predicted_rules = model(source.to(device))
            predicted_rules = torch.argmax(predicted_rules, dim=-1)
            predicted_rules = predicted_rules.cpu().long().numpy().tolist()

            lengths = (source != 0).sum(dim=-1).flatten().long().numpy().tolist()
            source = source.cpu().long().numpy().tolist()

            for char_indices, rule_indices, length, gold in zip(
                source, predicted_rules, lengths, true_unsandhied[i * batch_size :]
            ):
                # ----check predictions ----
                gold = " ".join(gold)
                gold = list(to_intern(gold))
                # if translit:
                #     gold = to_uni(gold)
                # gold = gold.split(" ")  # pretty stupid
                # logger.info(gold)
                char_indices = char_indices[:length]
                rule_indices = rule_indices[:length]

                chars = indexer.restore_indexed_sent(char_indices)
                rules = indexer.restore_indexed_rules(rule_indices)

                current_prediction = []
                rules_predicted = []  # check predicted rules
                previous_rule = None

                for char, rule in zip(chars, rules):
                    if rule == KEEP_RULE:
                        current_prediction.append(char)
                    elif rule == APPEND_SPACE_RULE:
                        current_prediction.append(char)
                        current_prediction.append(" ")

                    elif rule != previous_rule:
                        current_prediction.append(rule)
                    else:
                        pass

                    previous_rule = rule
                    # ----check predictions ----
                    rules_predicted.append((char, rule))

                if translit:
                    predicted_sent = "".join(current_prediction)
                    predictions.append(to_uni(predicted_sent))
                else:
                    predicted_sent = "".join(current_prediction)
                    predictions.append(predicted_sent)

                # ---- check predictions ----
                if current_prediction != gold:
                    wrong_predictions.append((chars, current_prediction, gold))
                    rules_used.append(rules_predicted)
                else:
                    logger.info("We actually got this right XD")
                # ---------------------------

        # if translit:
        #     predictions = list(map(to_uni, predictions))

        return predictions, wrong_predictions, rules_used


def output_wrong_predictions(
    wrong_predictions, rules_used, filename="t1_wrong_predictions.txt"
):
    with open(filename, "w", encoding="utf-8") as f:
        for (source, pred, gold), rules in zip(wrong_predictions, rules_used):
            f.write(f"\nSource:\t{list(source)}\n")
            f.write(f"Gold:\t{gold}\n")
            f.write(f"Pred:\t{pred}\n")
            f.write(f"Rules:\t{rules}\n")
            f.write(f"S:\t{''.join(source)}\n")
            f.write(f"G:\t{''.join(gold)}\n")
            f.write(f"P:\t{''.join(pred)}\n")
