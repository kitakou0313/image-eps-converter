from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import csv
import sys

if __name__ == "__main__":
    LOG_PATH = 'datas/graph-datas/uncased/sampled-for-ntgaed-clean-mlp/logs/events.out.tfevents.1675232493.f7efdf6531e5.66.0'
    OUTPUT_DIR_PATH = './'

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

    with open(OUTPUT_DIR_PATH+"/train.csv", 'w') as f:
        writer = csv.DictWriter(f, csv_header)
        writer.writeheader()
        writer.writerows(train_acc)

    with open(OUTPUT_DIR_PATH+"/test.csv", 'w') as f:
        writer = csv.DictWriter(f, csv_header)
        writer.writeheader()
        writer.writerows(test_acc)
