import torch

from tqdm import tqdm
from reconstruct_translit import reconstruct_unsandhied as reconstruct_translit
from reconstruct_uni import reconstruct_unsandhied
from uni2intern import internal_transliteration_to_unicode as to_uni

from sandhi_rules import KEEP_RULE, APPEND_SPACE_RULE


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
