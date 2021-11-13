import torch
import numpy as np

from tqdm import tqdm
from functools import partial
from sklearn.metrics import accuracy_score
from vocabulary import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN
from decoding import greedy_decoding, informed_decoding

from extract_rules import generate_stems_from_rules
from uni2intern import internal_transliteration_to_unicode as to_uni


def get_allowed_labels(raw_inputs, tag_encoder, rules):
    """
    Given an batch of input tokens, collects all possible stems and tokens
    from the dictionary

    raw_inputs: Inputs strings as nested list [[tokens] sentences]
    """
    allowed_stems = []
    allowed_tags = []

    for sent in raw_inputs:
        for token in sent:
            stems, tags = generate_stems_from_rules(token, rules)
            tags = [tag for tag in tags if tag in tag_encoder.classes_]

            allowed_stems.append(stems)
            allowed_tags.append(tag_encoder.transform(tags).tolist())

    return allowed_stems, allowed_tags


def evaluate_batch(
    batch, model, mode, tag_encoder, char2index, index2char, device, rules=None
):
    raw_inputs, inputs, stems, tags = batch
    stems, tags = stems.long(), tags.long()

    inputs = inputs.to(device)
    stems = stems.to(device)
    tags = tags.to(device)

    encoder = model["encoder"]
    decoder = model["decoder"]
    tag_classifier = model["tag_classifier"]

    encoder.eval()
    decoder.eval()
    tag_classifier.eval()

    if mode == "informed":
        allowed_stems, allowed_tags = get_allowed_labels(raw_inputs, tag_encoder, rules)

    # Get total number of tokens
    num_tokens = sum([len(sent) for sent in raw_inputs])

    # Encode input sentence
    encoded, char_embeddings, token_lengths = encoder(inputs)

    # Predict morphological tags
    pad_tag = tag_encoder.transform([PAD_TOKEN]).item()
    y_pred_tags = tag_classifier(encoded)  # shape (batch, timesteps, classes)
    assert y_pred_tags.shape[0] == num_tokens

    # For greedy decoding, we take the label
    # with highest prediction score
    if mode == "greedy":
        y_pred_tags = torch.argmax(y_pred_tags, dim=-1)  # shape (batch, timesteps)
        y_pred_tags = y_pred_tags.flatten().detach().cpu().numpy()

    # For informed decoding, we choose the allowed label
    # with highest prediction score
    elif mode == "informed":
        predicted_tags = []

        # Look at each token individually
        for scores, current_allowed_tags in zip(y_pred_tags, allowed_tags):
            # If we do not have any allowed tags, choose tag
            # with highest prediction score
            if len(current_allowed_tags) == 0:
                predicted_tags.append(torch.argmax(scores).item())

            # If we have exactly 1 allowed tag, choose this tag
            elif len(current_allowed_tags) == 1:
                predicted_tags.append(current_allowed_tags[0])

            # If we have multiple allowed tags, choose the tag
            # among the allowed tags with highest prediction score
            else:
                scores = scores[current_allowed_tags]
                best_score_index = torch.argmax(scores).item()
                best_tag = current_allowed_tags[best_score_index]
                predicted_tags.append(best_tag)

        y_pred_tags = np.array(predicted_tags)

    predicted_tags = tag_encoder.inverse_transform(y_pred_tags)

    # Predict stems
    if mode == "informed":
        predicted_stems = informed_decoding(
            decoder,
            encoded,
            char_embeddings,
            token_lengths,
            allowed_stems,
            char2index,
            index2char,
        )

    elif mode == "greedy":
        predicted_stems = greedy_decoding(
            decoder, encoded, char_embeddings, token_lengths, char2index
        )

    assert predicted_stems.shape[0] == num_tokens

    decoded_stems = []
    for char_indices in predicted_stems:
        current_stem = []
        for char_index in char_indices:
            # Convert current index to character string
            character = index2char[char_index.item()]

            # If we encounte end of sequence, finish the current stem
            if character == EOS_TOKEN:
                break
            # Otherwise, save current character
            else:
                current_stem.append(character)
        decoded_stems.append("".join(current_stem))

    prediction_pointer = 0
    batch_tag_predictions = []
    batch_stem_predictions = []
    for sent in raw_inputs:
        sent_predicted_tags = []
        sent_predicted_stems = []
        for token in sent:
            sent_predicted_tags.append(predicted_tags[prediction_pointer])
            sent_predicted_stems.append(decoded_stems[prediction_pointer])
            prediction_pointer += 1

        batch_tag_predictions.append(sent_predicted_tags)
        batch_stem_predictions.append(sent_predicted_stems)

    assert prediction_pointer == len(predicted_tags) == len(decoded_stems)

    # Save predicted tags and stems for current minibatch
    batch_predictions = zip(batch_tag_predictions, batch_stem_predictions)
    batch_predictions = list(batch_predictions)

    return batch_predictions


def evaluate_model(
    model,
    dataloader,
    tag_encoder,
    char2index,
    index2char,
    device,
    mode="greedy",
    rules=None,
):
    # Check arguments
    if mode not in ["greedy", "informed"]:
        raise ValueError(f"Unknown evaluation mode: {mode}")
    if mode == "informed" and rules is None:
        raise RuntimeError("Evaluation mode is 'informed' but no dictionary given")

    predictions = []
    get_predictions_from_batch = partial(
        evaluate_batch,
        model=model,
        mode=mode,
        tag_encoder=tag_encoder,
        device=device,
        char2index=char2index,
        index2char=index2char,
        rules=rules,
    )

    with torch.no_grad():
        for batch in tqdm(dataloader):
            batch_predictions = get_predictions_from_batch(batch)
            predictions.extend(batch_predictions)

    return predictions


def print_metrics(predictions, ground_truth):
    tags_true, tags_pred, stems_true, stems_pred = [], [], [], []
    assert len(predictions) == len(ground_truth)

    for y_true, y_pred in zip(ground_truth, predictions):
        _, labels = y_true
        stems, tags = zip(*labels)
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
