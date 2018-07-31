from numpy import mean, array, float32


def get_batch_accuracy(predictions, labels):
    accuracy = []
    for index, gt_label in enumerate(labels):
        pred = predictions[index]
        totol_count = len(gt_label)
        correct_count = 0
        try:
            for i, tmp in enumerate(gt_label):
                if tmp == pred[i]:
                    correct_count += 1
        except IndexError:
            continue
        finally:
            try:
                accuracy.append(correct_count / totol_count)
            except ZeroDivisionError:
                if len(pred) == 0:
                    accuracy.append(1)
                else:
                    accuracy.append(0)
    return accuracy


def calculate_mean_accuracy(accuracy: list) -> float:
    return mean(array(accuracy).astype(float32), axis=0)
