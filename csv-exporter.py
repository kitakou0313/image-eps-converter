from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv
import os

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", required=True, type=str,
                        help="path to tensorboard log data")
    parser.add_argument("--csv_dir", required=True, type=str,
                        help="path to dir to save csv data")
    args = parser.parse_args()

    LOG_PATH = args.log_path
    OUTPUT_DIR_PATH = args.csv_dir

    event_acc = EventAccumulator(LOG_PATH, size_guidance={'scalars': 0})
    event_acc.Reload()  # ログファイルのサイズによっては非常に時間がかかる

    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]

    train_acc: list[dict] = []
    test_acc: list[dict] = []

    for ind, acc in enumerate(scalars["Train/acc"]):
        train_acc.append({"Value": acc, "Step": ind+1})

    for ind, acc in enumerate(scalars["Validation/acc"]):
        test_acc.append({"Value": acc, "Step": ind+1})

    csv_header = ("Step", "Value")

    with open(os.path.join(OUTPUT_DIR_PATH, "train.csv"), 'w') as f:
        writer = csv.DictWriter(f, csv_header)
        writer.writeheader()
        writer.writerows(train_acc)

    with open(os.path.join(OUTPUT_DIR_PATH, "test.csv"), 'w') as f:
        writer = csv.DictWriter(f, csv_header)
        writer.writeheader()
        writer.writerows(test_acc)
