from numpy import mean, array, float32


def get_batch_accuracy(predictions, labels) -> list:
    accuracy = []
    for index, gt_label in enumerate(labels):
        pred = predictions[index]
        acc = get_accuracy(pred, gt_label)
        accuracy.append(acc)
    return accuracy


def get_accuracy(prediction, gt_label) -> float:
    total_count = len(gt_label)
    pred_count = len(prediction)
    count = total_count if total_count < pred_count else pred_count
    if total_count == 0 and pred_count == 0:
        return 1
    if total_count == 0 and pred_count != 0:
        return 0
    correct_count = 0
    try:
        for i in range(count):
            if gt_label[i] == prediction[i]:
                correct_count += 1
    except IndexError:
        pass
    finally:
        return correct_count / total_count


def calculate_array_mean(accuracy: list) -> float:
    return mean(array(accuracy).astype(float32), axis=0)
