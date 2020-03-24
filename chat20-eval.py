# !/usr/bin/python3
# -*- coding: utf-8 -*-
from argparse import ArgumentParser
from sys import stderr
import pandas as pd
from sklearn.metrics import f1_score

EV_OUT = "evaluation.prototext"


def parse_input() -> tuple:
    """ read the input files to truth and predictions and check input validity
    :return: a tuple with   [0] a dataframe with the user/channel pairs and the true and predicted subscriber status,
                            [1] the path to the output directory.
    """
    parser = ArgumentParser("A script to evaluate the performance of a ChAT20 submission.")
    parser.add_argument("-p", "--predictions", help="path to the directory with the predicted predictions.csv")
    parser.add_argument("-t", "--truth", help="path to the directory with the true truth.csv")
    parser.add_argument("-o", "--output", help="path to the directory to write the results to")
    args = parser.parse_args()

    targets = pd.read_csv(f"{args.truth}/truth.csv")
    predictions = pd.read_csv(f"{args.predictions}/predictions.csv")

    if not len(targets) == len(predictions):
        print("Invalid output file, predictions for some users are missing.", file=stderr)

    merged = pd.merge(targets, predictions, how="inner", on=["channel", "user"], suffixes=("_target", "_pred"))

    if not len(merged) == len(targets):
        print("Invalid output file, predictions for some users are missing.", file=stderr)

    return merged, args.output


def write_output(filename, k, v):
    """
    print() and write a given measurement to the indicated output file
    :param filename: full path of the file, where to write to
    :param k: the name of the metric
    :param v: the value of the metric
    :return: None
    """
    line = 'measure{{\n  key: "{}"\n  value: "{}"\n}}\n'.format(k, str(v))
    print(line)
    open(filename, "a").write(line)


if __name__ == "__main__":
    """
    This is the evaluator for the ChAT Discovery Challenge at the ECML/PKDD 2020
    It outputs the F1-score of the predictions provided

    For more information visit: 
      - https://events.professor-x.de/dc-ecmlpkdd-2020/

    Please send any requests or remarks to:
      - Konstantin Kobs, kobs@informatik.uni-wuerzburg.de
    """

    results, output_dir = parse_input()
    score = f1_score(results["subscribed_target"], results["subscribed_pred"], average="binary", pos_label=False)
    write_output("{}/{}".format(output_dir, EV_OUT), "f1", score)
