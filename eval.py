import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from sklearn.metrics import confusion_matrix

def evaluate(dataset_val, model, config):
    gt_classes = [0 for _ in range(config.num_classes)]
    positive_classes = [0 for _ in range(config.num_classes)]
    true_positive_classes = [0 for _ in range(config.num_classes)]
    val_total_correct = 0
    val_total_seen = 0

    for step_id, data in enumerate(dataset_val.create_tuple_iterator()):
        # print("start")
        if step_id % 50 == 0:
            print(str(step_id) + ' / ' + str(config.val_steps))
        logits, _ = model(data)
        # logits = model(data)
        logits = logits
        logits = ops.Reshape()(logits, (-1, logits.shape[-1]))
        labels = ops.Reshape()(data[4 * config.num_layers + 1], (-1,))

        pred = logits.argmax(1).asnumpy()
        labels = labels.asnumpy()
        if config.ignored_label_inds:
            invalid_idx = np.array([], np.int32)
            for ign_label in config.ignored_label_inds:
                invalid_idx = np.append(invalid_idx, np.where(labels == ign_label)[0])
            labels_valid = np.delete(labels, invalid_idx)
            # labels_valid = labels_valid - 1
            pred_valid = np.delete(pred, invalid_idx)
        else:
            pred_valid = pred
            labels_valid = labels

        correct = np.sum(pred_valid == labels_valid)
        val_total_correct += correct
        val_total_seen += len(labels_valid)

        conf_matrix = confusion_matrix(labels_valid, pred_valid, np.arange(0, config.num_classes, 1))
        gt_classes += np.sum(conf_matrix, axis=1)
        positive_classes += np.sum(conf_matrix, axis=0)
        true_positive_classes += np.diagonal(conf_matrix)
        # print("end")

    iou_list = []
    for n in range(0, config.num_classes):
        union = gt_classes[n] + positive_classes[n] - true_positive_classes[n]
        if union == 0: iou = 0.0
        else: iou = true_positive_classes[n] / float(union)
        iou_list.append(iou)
    mean_iou = sum(iou_list) / float(config.num_classes)

    print('eval accuracy: {}'.format(val_total_correct / float(val_total_seen)))
    print('mean IOU:{}'.format(mean_iou))

    mean_iou = 100 * mean_iou
    print('Mean IoU = {:.1f}%'.format(mean_iou))
    s = '{:5.2f} | '.format(mean_iou)
    for IoU in iou_list:
        s += '{:5.2f} '.format(100 * IoU)
    print('-' * len(s))
    print(s)
    print('-' * len(s) + '\n')
    return mean_iou