#!/usr/bin/env python3

# Scoring Program for Word Segmentation and Morphological Parsing

import json
import os
import sys
from sys import argv
import numpy as np


root_dir = "../"
default_solution_dir = root_dir + "reference_data"
default_prediction_dir = root_dir + "result_submission"
default_score_dir = root_dir + "scores"


def is_list_for_task_1(lst):
    """
    Checks if the given list is of the format: list of list of strings
    Input: List of all predictions for task 1
    Returns True/False accordingly
    """
    return isinstance(lst, list) and all(
        isinstance(lt, list) and all(isinstance(elem, str) for elem in lt) for lt in lst
    )


def is_list_for_task_2(lst):
    """
    Checks if the given list is of the format: list of list of list of (string, string)
    Input: List of all predictions for task 2
    Returns True/False accordingly
    """
    return isinstance(lst, list) and all(
        isinstance(lt, list)
        and all(
            isinstance(items, list)
            and (len(items) == 2)
            and all(isinstance(item, str) for item in items)
            for items in lt
        )
        for lt in lst
    )


def is_list_for_task_3(lst):
    """
    Checks if the given list is of the format: list of list of list of (string, string, string)
    Input: List of all predictions for task 3
    Returns True/False accordingly
    """
    return isinstance(lst, list) and all(
        isinstance(lt, list)
        and all(
            isinstance(items, list)
            and (len(items) == 3)
            and all(isinstance(item, str) for item in items)
            for items in lt
        )
        for lt in lst
    )


def evaluate_task1(sol, pred):
    """
    Returns the recall, precision, task 1 score, and accuracy for the given sentence
    Input: solution list of strings, predictions list of strings
    Outpu: recall float, precision float, task1_score float, accuracy bool
    """
    matches = 0
    t1_score = 0.0
    min_len = min(
        len(sol), len(pred)
    )  # Temporarily taking the minimum length of the two
    # Should handle cases when the segment is predicted later or ahead of its actual position or if it is joint with another segment
    for i in range(min_len):
        if pred[i] == sol[i]:
            matches += 1
            t1_score += (
                1.0  # A value of 1.0 is assigned to the correct match of the string
            )

    t1_score = t1_score / (len(sol))  # Mean of the observed task scores

    recall = matches / len(sol)
    precision = matches / len(pred)

    return (recall, precision, t1_score, (pred == sol))


def evaluate_task2(sol_list, pred_list):
    """
    Returns the partial recall, partial precision, full recall, full precision, task 2 score, and accuracy for the given sentence
    Input: solution list of list of 3 strings, predictions list of list of 3 strings
    Output: partial recall float, partial precision float, full recall float, full precision float, task 2 score float, accuracy bool
    """
    pred = [
        tuple((sublist[0], sublist[1])) for sublist in pred_list if (len(sublist) == 2)
    ]  # convert the prediction to a list of list of tuples with 2 strings
    sol = [
        tuple((sublist[0], sublist[1])) for sublist in sol_list
    ]  # convert the solution to a list of list of tuples with 2 strings

    lemma_matches = 0
    morph_matches = 0
    complete_matches = 0
    t2_score = 0.0
    min_len = min(len(sol), len(pred))
    for i in range(min_len):
        if not (
            len(pred[i]) == 2
        ):  # every word in input requires a prediction of 2 entities (stem and morph category).
            continue
        if (
            pred[i] == sol[i]
        ):  # for complete matches. assign 2.0 as task score because there are two predictions
            complete_matches += 1
            t2_score += 2.0
        elif pred[i][0] == sol[i][0]:  # for partial - stem matches
            lemma_matches += 1
            t2_score += 1.0
        elif pred[i][1] == sol[i][1]:  # for partial - morph category matches
            morph_matches += 1
            t2_score += 1.0

    t2_score = t2_score / (
        2.0 * len(sol)
    )  # calculating the final task score for the current prediction and normalizing it to a scale of 0-1

    partial_matches = (
        lemma_matches + morph_matches
    )  # partial matches are calculated. if required, we may modify the scoring to include these as well
    partial_recall = (partial_matches) / len(sol)
    partial_precision = (partial_matches) / len(pred)
    full_recall = complete_matches / len(sol)
    full_precision = complete_matches / len(pred)

    return (
        partial_recall,
        partial_precision,
        full_recall,
        full_precision,
        t2_score,
        (pred == sol),
    )


def evaluate_task3(sol_list, pred_list):
    """
    Returns the partial recall, partial precision, full recall, full precision, task 3 score, and accuracy for the given sentence
    Input: solution list of list of 3 strings, predictions list of list of 3 strings
    Output: partial recall float, partial precision float, full recall float, full precision float, task 3 score float, accuracy bool
    """
    pred = [
        tuple((sublist[0], sublist[1], sublist[2]))
        for sublist in pred_list
        if (len(sublist) == 3)
    ]  # convert the prediction to a list of list of tuples with 3 strings
    sol = [
        tuple((sublist[0], sublist[1], sublist[2])) for sublist in sol_list
    ]  # convert the solution to a list of list of tuples with 3 strings

    word_matches = 0
    lemma_matches = 0
    morph_matches = 0
    complete_matches = 0
    t3_score = 0.0
    min_len = min(len(sol), len(pred))
    for i in range(min_len):
        if not (
            len(pred[i]) == 3
        ):  # every word in input requires a prediction of 3 entities (word, stem and morph category).
            continue
        if (
            pred[i] == sol[i]
        ):  # for complete matches. assign 3.0 as task score because there are three predictions
            complete_matches += 1
            t3_score += 3.0
        else:
            if pred[i][0] == sol[i][0]:  # for partial - word matches
                word_matches += 1
                t3_score += 1.0
            elif pred[i][1] == sol[i][1]:  # for partial - stem matches
                lemma_matches += 1
                t3_score += 1.0
            elif pred[i][2] == sol[i][2]:  # for partial - morph category matches
                morph_matches += 1
                t3_score += 1.0

    t3_score = t3_score / (
        3.0 * len(sol)
    )  # calculating the final task score for the current prediction and normalizing it to a scale of 0-1

    partial_matches = (
        word_matches + lemma_matches + morph_matches
    )  # partial matches are calculated. if required, we may modify the scoring to include these as well
    partial_recall = (partial_matches) / len(sol)
    partial_precision = (partial_matches) / len(pred)
    full_recall = complete_matches / len(sol)
    full_precision = complete_matches / len(pred)

    return (
        partial_recall,
        partial_precision,
        full_recall,
        full_precision,
        t3_score,
        (pred == sol),
    )


def evaluate(solution, prediction, task_id):
    """
    Returns the macro averaged - precision, recall, f1_score, and task score - as a dict
    Input: solution list of test reference data, prediction list of predictions and task id
    Output: Average Precision, Average Recall, F1 Score, Average Task score
    """
    scores = {}
    if len(prediction) == 0:
        print(
            "\nWrong file format for prediction in task "
            + task_id
            + ". Refer instructions for the expected format.",
            file=sys.stderr,
        )
        return scores
    if (
        ((task_id == "t1") and (not (is_list_for_task_1(prediction))))
        or ((task_id == "t2") and (not (is_list_for_task_2(prediction))))
        or ((task_id == "t3") and (not (is_list_for_task_3(prediction))))
    ):
        print(
            "\nWarning: Wrong format for some predictions in task "
            + task_id
            + ". Refer instructions for the expected format.",
            file=sys.stderr,
        )

    print("\nEvaluating for task " + task_id)
    precisions = []
    recalls = []
    accuracies = []
    task_scores = []
    for pred, sol in zip(
        prediction, solution
    ):  # iterate through the prediction and solution list
        if (
            task_id == "t2"
        ):  # to convert the list of lists of values to list of tuples (2)
            try:
                (
                    part_rec,
                    part_prec,
                    full_rec,
                    full_prec,
                    task_score,
                    accuracy,
                ) = evaluate_task2(sol, pred)
            except Exception as e:
                (part_rec, part_prec, full_rec, full_prec, task_score, accuracy) = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    False,
                )  # return zeroes if caught in any exception
        elif (
            task_id == "t3"
        ):  # to convert the list of lists of values to list of tuples (3)
            try:
                (
                    part_rec,
                    part_prec,
                    full_rec,
                    full_prec,
                    task_score,
                    accuracy,
                ) = evaluate_task3(sol, pred)
            except Exception as e:
                (part_rec, part_prec, full_rec, full_prec, task_score, accuracy) = (
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    False,
                )  # return zeroes if caught in any exception
        else:  # use the same list
            try:
                (full_rec, full_prec, task_score, accuracy) = evaluate_task1(sol, pred)
            except Exception as e:
                (full_rec, full_prec, task_score, accuracy) = (
                    0.0,
                    0.0,
                    0.0,
                    False,
                )  # return zeroes if caught in any exception

        precisions.append(full_prec)  # accumulate precisions
        recalls.append(full_rec)  # accumulate recall values
        task_scores.append(task_score)  # accumulate task scores

        if (
            accuracy
        ):  # condition for accuracy to accumulate 1.0 for accurate prediction and 0.0 for inaccurate prediction
            accuracies.append(1.0)
        else:
            accuracies.append(0.0)

    if not (len(prediction) == len(solution)):
        print(
            str((len(solution) - len(prediction)))
            + " of expected predictions are missing in task "
            + task_id,
            file=sys.stderr,
        )
        #        return scores                                        # Either return empty scores or calculate these for the given predictions
        for i in range(
            (len(solution) - len(prediction))
        ):  # for the remaining solutions, assign zeroes
            precisions.append(0.0)
            recalls.append(0.0)
            task_scores.append(0.0)
            accuracies.append(0.0)

    avg_prec = np.mean(precisions) * 100.0  # calculating average precision
    avg_recall = np.mean(recalls) * 100.0  # calculating average recall
    f1_score = (
        2 * avg_prec * avg_recall / (avg_prec + avg_recall)
        if (not ((avg_prec + avg_recall) == 0))
        else 0.0
    )  # calculating f1 score
    avg_accuracies = np.mean(accuracies)  # calculating accuracy scores
    avg_task_scores = np.mean(task_scores) * 100.0

    if task_id == "t1":
        scores = {
            "task_1_precision": avg_prec,
            "task_1_recall": avg_recall,
            "task_1_f1score": f1_score,
            "task_1_tscore": avg_task_scores,
        }
    elif task_id == "t2":
        scores = {
            "task_2_precision": avg_prec,
            "task_2_recall": avg_recall,
            "task_2_f1score": f1_score,
            "task_2_tscore": avg_task_scores,
        }
    elif task_id == "t3":
        scores = {
            "task_3_precision": avg_prec,
            "task_3_recall": avg_recall,
            "task_3_f1score": f1_score,
            "task_3_tscore": avg_task_scores,
        }

    print_scores(scores)

    return scores


def calculate_scores(prediction_file, reference_file, task_id):
    """
    Builds score for the prediction and saves it in file
    Input: file with prediction results, file with reference and task id
    Returns the results from evaluation
    """
    print("Reading prediction")
    with open(prediction_file, encoding="utf-8") as f:
        try:
            prediction = list(json.load(f))  # load prediction values
        except Exception as e:
            print(
                "Could not load the predictions file. Please check the format of the"
                " predictions.",
                file=sys.stderr,
            )
            print("Exception: " + str(e), file=sys.stderr)
            raise FileNotLoaded
    #    with open(duration_file) as f:
    #        duration = json.load(f).get('duration', -1)                                  # load duration - removed duration for prediction
    with open(reference_file, encoding="utf-8") as f:
        truth = list(json.load(f))  # load reference values

    return evaluate(
        truth, prediction, task_id
    )  # get evaluation results as macro-averaged - precision, recall, f1score, task score


if __name__ == "__main__":

    # INPUT/OUTPUT: Get input and output directory names
    if len(argv) == 1:  # Use the default data directories if no arguments are provided
        reference_dir = default_solution_dir
        prediction_dir = default_prediction_dir
        score_dir = default_score_dir
    elif len(argv) == 3:  # The current default configuration of Codalab
        reference_dir = os.path.join(argv[1], "ref")
        prediction_dir = os.path.join(argv[1], "res")
        score_dir = argv[2]
    elif len(argv) == 4:
        reference_dir = argv[1]
        prediction_dir = argv[2]
        score_dir = argv[3]
    else:
        print("\n*** WRONG NUMBER OF ARGUMENTS ***\n\n", file=sys.stderr)
        exit(1)

    if not (
        os.path.isdir(score_dir)
    ):  # Create the score directory, if it does not already exist
        os.mkdir(score_dir)

    scores_entries = {}

    print("Checking predictions for Task 1 - Word Segmentation")
    if os.path.isfile(os.path.join(prediction_dir, "task1_predictions.json")):  # Task 1
        print("Predictions for Task 1 - Word Segmentation Found")
        try:
            task_1_scores_entries = calculate_scores(
                os.path.join(prediction_dir, "task1_predictions.json"),
                os.path.join(reference_dir, "task1_reference.json"),
                "t1",
            )
        except FileNotLoaded:
            task_1_scores_entries = {}
        if not (
            len(task_1_scores_entries.keys()) == 0
        ):  # Update if scores are obtained. Else throw error
            scores_entries.update(task_1_scores_entries)
        else:
            print(
                "Scores could not be calculated for Task 1. No predictions found."
                " Either there is no prediction or the format of the prediction is"
                " wrong.",
                file=sys.stderr,
            )
    else:
        print("Predictions for Task 1 - Word Segmentation Not Found")

    print("\nChecking predictions for Task 2 - Morphological Parsing")
    if os.path.isfile(os.path.join(prediction_dir, "task2_predictions.json")):  # Task 2
        print("Predictions for Task 2 - Morphological Parsing Found")
        try:
            task_2_scores_entries = calculate_scores(
                os.path.join(prediction_dir, "task2_predictions.json"),
                os.path.join(reference_dir, "task2_reference.json"),
                "t2",
            )
        except FileNotLoaded:
            task_2_scores_entries = {}
        if not (
            len(task_2_scores_entries.keys()) == 0
        ):  # Update if scores are obtained. Else throw error
            scores_entries.update(task_2_scores_entries)
        else:
            print(
                "Scores could not be calculated for Task 2. No predictions found."
                " Either there is no prediction or the format of the prediction is"
                " wrong.",
                file=sys.stderr,
            )
    else:
        print("Predictions for Task 2 - Morphological Parsing Not Found")

    print(
        "\nChecking predictions for Task 3 - Combined Word Segmentation and"
        " Morphological Parsing"
    )
    if os.path.isfile(os.path.join(prediction_dir, "task3_predictions.json")):  # Task 3
        print(
            "Predictions for Task 3 - Combined Word Segmentation and Morphological"
            " Parsing Found"
        )
        try:
            task_3_scores_entries = calculate_scores(
                os.path.join(prediction_dir, "task3_predictions.json"),
                os.path.join(reference_dir, "task3_reference.json"),
                "t3",
            )
        except:
            task_3_scores_entries = {}
        if not (
            len(task_3_scores_entries.keys()) == 0
        ):  # Update if scores are obtained. Else throw error
            scores_entries.update(task_3_scores_entries)
        else:
            print(
                "Scores could not be calculated for Task 3. No predictions found."
                " Either there is no prediction or the format of the prediction is"
                " wrong.",
                file=sys.stderr,
            )
    else:
        print(
            "Predictions for Task 3 - Combined Word Segmentation and Morphological"
            " Parsing Not Found"
        )

    if len(scores_entries.keys()) == 0:
        print(
            "\nScores could not be calculated for any of the tasks. Please check the"
            " format of the predictions file(s)",
            file=sys.stderr,
        )
        exit(1)
    else:
        scores_string = ""
        for key in scores_entries.keys():
            scores_string = (
                scores_string + key + ":" + str(scores_entries.get(key)) + "\n"
            )

        with open((os.path.join(score_dir, "scores.txt")), "w") as score_f:
            score_f.write(scores_string)
        print("\nFinished writing to scores.txt")


def print_scores(scores):
    print("-----------")
    for metric, score in scores.items():
        print(f"{metric}:\t{score:.2f}")
