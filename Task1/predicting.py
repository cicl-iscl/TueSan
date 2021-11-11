import torch

from tqdm import tqdm
from reconstruct_translit import reconstruct_unsandhied as reconstruct_translit
from reconstruct_uni import reconstruct_unsandhied
from uni2intern import internal_transliteration_to_unicode as to_uni


def make_predictions(model, eval_dataloader, eval_dataset, device, translit=False):

    split_predictions = []

    with torch.no_grad():
        for inputs in tqdm(eval_dataloader):
            y_pred = model(inputs.to(device)).detach().cpu()
            y_pred = torch.round(y_pred).bool()

            for sample in y_pred:
                split_indices = torch.arange(0, len(sample))[sample]
                split_indices = split_indices.numpy()
                split_predictions.append(split_indices.tolist())

    predictions = []
    true_unsandhied = []

    for sentence, split_indices in zip(eval_dataset, split_predictions):
        sandhied = sentence["sandhied"]
        allowed_words = sentence["allowed_words"]

        span_starts = [0] + split_indices
        span_ends = split_indices + [None]
        tokens = [sandhied[i:j] for i, j in zip(span_starts, span_ends)]

        if translit:
            predicted_unsandhied = reconstruct_translit(tokens, allowed_words)
            predictions.append(
                [to_uni(x) for x in predicted_unsandhied]
            )  # predictions in unicode
            true_unsandhied.append(to_uni(sentence["unsandhied"]))
        else:
            predicted_unsandhied = reconstruct_unsandhied(tokens, allowed_words)
            predictions.append(predicted_unsandhied)
            true_unsandhied.append(sentence["unsandhied"])

    return predictions, true_unsandhied, split_predictions
