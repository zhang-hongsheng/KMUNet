import numpy as np


def calculate_metrics(pred, target, num_classes):

    iou_list = []
    accuracy_list = []
    dice_list = []
    recall_list = []
    precision_list = []
    specificity_list = []

    for class_id in range(0, num_classes):
        pred_class = (pred == class_id).cpu().numpy()
        target_class = (target == class_id).cpu().numpy()

        true_positives = np.sum(np.logical_and(pred_class == 1, target_class == 1))
        false_positives = np.sum(np.logical_and(pred_class == 1, target_class == 0))
        false_negatives = np.sum(np.logical_and(pred_class == 0, target_class == 1))
        true_negatives = np.sum(np.logical_and(pred_class == 0, target_class == 0))

        intersection = true_positives
        union = true_positives + false_positives + false_negatives
        iou = intersection / (union + 1e-7)
        iou_list.append(iou)

        correct = np.sum(pred_class == target_class)
        total_pixels = np.prod(target_class.shape)
        accuracy = correct / total_pixels

        accuracy_list.append(accuracy)

        dice = 2 * true_positives / (2 * true_positives + false_positives + false_negatives + 1e-7)
        dice_list.append(dice)

        recall = true_positives / (true_positives + false_negatives + 1e-7)
        recall_list.append(recall)

        precision = true_positives / (true_positives + false_positives + 1e-7)
        precision_list.append(precision)

        specificity = true_negatives / (true_negatives + false_positives + 1e-7)
        specificity_list.append(specificity)


    return {
        'iou': iou_list,
        'accuracy': accuracy_list,
        'dice': dice_list,
        'recall': recall_list,
        'precision': precision_list,
        'specificity': specificity_list,

    }
