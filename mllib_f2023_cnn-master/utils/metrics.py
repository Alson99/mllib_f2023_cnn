def accuracy(predicted_classes, ground_truth):
    """
        Вычисление точности:
            accuracy = sum( predicted_class == ground_truth ) / N, где N - размер набора данных
        # реализуйте подсчет accuracy
    """
    assert len(predicted_classes) == len(ground_truth), "Input sizes must be the same"

    correct_predictions = sum(p == gt for p, gt in zip(predicted_classes, ground_truth))
    accuracy = correct_predictions / len(ground_truth)
    return accuracy


def balanced_accuracy(predicted_classes, ground_truth, num_classes):
    """
        Вычисление точности:
            balanced accuracy = sum( TP_i / N_i ) / N, где
                TP_i - кол-во изображений класса i, для которых предсказан класс i
                N_i - количество изображений набора данных класса i
                N - количество классов в наборе данных
        # реализуйте подсчет balanced accuracy
    """
    assert len(predicted_classes) == len(ground_truth), "Input sizes must be the same"

    class_counts = [0] * num_classes
    true_positives = [0] * num_classes

    for predicted, gt in zip(predicted_classes, ground_truth):
        class_counts[gt] += 1
        if predicted == gt:
            true_positives[gt] += 1

    balanced_accuracy = sum(
        tp / count if count != 0 else 0 for tp, count in zip(true_positives, class_counts)) / num_classes
    return balanced_accuracy
