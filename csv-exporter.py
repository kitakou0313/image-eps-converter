from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

if __name__ == "__main__":
    LOG_PATH = 'datas/graph-datas/uncased/fixed-for-chap-7/logs/events.out.tfevents.1674874989.e69379b7ec54.59.0'

    event_acc = EventAccumulator(LOG_PATH, size_guidance={'scalars': 0})
    event_acc.Reload()  # ログファイルのサイズによっては非常に時間がかかる

    scalars = {}
    for tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        scalars[tag] = [event.value for event in events]

    train_acc = {"Value": [], "Step": []}
    test_acc = {"Value": [], "Step": []}

    for ind, acc in enumerate(scalars["Train/acc"]):
        train_acc["Step"].append(ind+1)
        train_acc["Value"].append(acc)

    for ind, acc in enumerate(scalars["Validation/acc"]):
        test_acc["Step"].append(ind+1)
        test_acc["Value"].append(acc)

    pass
