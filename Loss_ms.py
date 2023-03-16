import numpy as np
import mindspore as ms
from mindspore import numpy as msnp
from mindspore import nn, ops, Tensor, Parameter
from helper_tool import ConfigS3DIS as cfg
from HybridCR_ms import Gather


def valid_select(labels, valid_label_inds):
    labels = ops.Reshape()(labels, (-1,))
    labels = labels.asnumpy()
    valid_idx = np.array([], np.int32)
    for valid_label in valid_label_inds:
        valid_idx = np.append(valid_idx, np.where(labels == valid_label)[0])
    return Tensor(valid_idx, ms.int32)

def global_embedding(logits, labels, embedding, num_classes, ignored_label_inds, confidence=0.95):
    batch_size, num_points = labels.shape
    mask0 = msnp.ones((batch_size * num_points, ), ms.int32)
    mask1 = msnp.zeros((batch_size * num_points, ), ms.int32)
    idx = [i for i in range(num_classes)]
    valid_idx = valid_select(labels, idx)
    if valid_idx.size:
        mask0[valid_idx] = 0
        mask1[valid_idx] = 1

    logits = ops.Reshape()(logits, (-1, logits.shape[-1]))
    logits = nn.Softmax(axis=-1)(logits)
    threshold = msnp.full(logits.shape, confidence, ms.float32)
    label = msnp.full(logits.shape, 2, ms.int32)
    unlabel = msnp.full(logits.shape, 0, ms.int32)
    pseudo_labels = msnp.where(msnp.greater_equal(logits, threshold), label, unlabel)
    pseudo_labels = ops.Concat(-1)((pseudo_labels, msnp.ones((pseudo_labels.shape[0], 1), ms.int32)))
    pseudo_labels = pseudo_labels.argmax(1)
    pseudo_labels = pseudo_labels * mask0
    labeled_labels = ops.Reshape()(labels, (-1,)) * mask1
    all_labels = pseudo_labels + labeled_labels
    valid_idx = valid_select(all_labels, idx)

    class_embedding = msnp.zeros((1, 32), ms.float32)
    embedding = ops.Reshape()(embedding, (-1, embedding.shape[-1]))
    for i in range(num_classes):
        valid_class_idx = valid_select(all_labels, [i])
        if valid_class_idx.size:
            valid_embedding = ops.Gather()(embedding, valid_class_idx, 0)
            valid_embedding = ops.ReduceMean(keep_dims=True)(valid_embedding, 0)
        else: valid_embedding = msnp.zeros((1, 32), ms.float32)
        class_embedding = ops.Concat(0)((class_embedding, valid_embedding))
    class_embedding = class_embedding[1:, :]

    return valid_idx, class_embedding

class Loss(nn.Cell):
    def __init__(self, batch_size, num_classes, num_points, lamda=1):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_points = num_points
        self.lamda = lamda

        self.N = self.batch_size * self.num_points
        self.mask = Parameter(msnp.zeros((self.N * 2, ), ms.int32), name="mask", requires_grad=False)
        self.onehot = nn.OneHot(depth=self.num_classes)
        self.ce_loss = nn.SoftmaxCrossEntropyWithLogits()
        self.softmax = nn.Softmax(axis=-1)

        self.range = nn.Range(0, self.N)
        self.cra_loss = Contrastive_Loss(self.N)
        self.lcl_loss = Contrastive_Loss(self.N)
        self.gcl_loss = Contrastive_Loss(self.num_classes)
        
    def construct(self, logits, labels, embedding, valid_idx, class_embedding, neigh_idx, pre_cal_weights):
        if valid_idx.size: self.mask[valid_idx] = 1

        # self-distillation loss
        logits = self.softmax(logits)
        p1 = logits[:self.batch_size, :, :]
        p2 = logits[self.batch_size:, :, :]
        q = (p1 + p2) / 2 + 1e-4
        loss_js = ops.ReduceMean()(p1 * ops.Log()(p1/q+1e-4) + p2 * ops.Log()(p2/q+1e-4))
        # ops.Print()(loss_js)

        # segmentation loss
        logits = ops.Reshape()(logits, (-1, logits.shape[-1]))
        labels = ops.Reshape()(labels, (-1,))
        one_hot_labels = self.onehot(labels)
        weights = ops.ReduceSum()(pre_cal_weights * one_hot_labels, 1)
        unweighted_losses = self.ce_loss(logits, one_hot_labels)
        weighted_losses = unweighted_losses * weights
        weighted_losses = weighted_losses * self.mask
        loss_ce = ops.ReduceMean()(weighted_losses)
        # ops.Print()(loss_ce)

        # contrastive loss
        embedding1 = embedding[:self.batch_size, :, :]
        embedding2 = embedding[self.batch_size:, :, :]
        G_embedding2 = Gather()(embedding2.transpose((0, 2, 1)), neigh_idx)
        # print("G_embedding2.shape: ", G_embedding2.shape)
        G_embedding2 = ops.ReduceMean(keep_dims=True)(G_embedding2, -1).squeeze(-1).transpose((0, 2, 1))
        # print("G_embedding2.shape: ", G_embedding2.shape)
        embedding = msnp.multiply(ops.Reshape()(self.mask, (-1, 1)), ops.Reshape()(embedding, (-1, embedding.shape[-1])))
        labeled_labels = labels * self.mask
        loss_cra = self.cra_loss(embedding1, embedding2, self.range())
        loss_lcl = self.lcl_loss(embedding1, G_embedding2, self.range())
        loss_gcl = self.gcl_loss(embedding, class_embedding, labeled_labels)
        loss_cr = self.lamda * (loss_cra + loss_lcl + loss_gcl)
        # ops.Print()(loss_cr)
        
        return loss_ce + loss_js + loss_cr

class Contrastive_Loss(nn.Cell):
    def __init__(self, dimension, temperature=1):
        super(Contrastive_Loss, self).__init__()
        self.LARGE_NUM = 1e9
        self.dimension = dimension
        self.temperature = temperature
        self.onehot = nn.OneHot(depth=self.dimension)

    def construct(self, embedding1, embedding2, mask):
        embedding1  = ops.L2Normalize(axis=1)(embedding1)
        embedding2 = ops.L2Normalize(axis=1)(embedding2)
        embedding1 = ops.Reshape()(embedding1, (-1, embedding1.shape[-1]))
        embedding2 = ops.Reshape()(embedding2, (-1, embedding2.shape[-1]))
        similar = ops.MatMul(transpose_b=True)(embedding1, embedding2) / self.temperature

        mask = self.onehot(mask)
        positive_similar = ops.ReduceSum()(similar * mask, 1)
        negative_similar = ops.ReduceSum()(ops.Exp()(similar - mask * self.LARGE_NUM), 1)
        loss = -1.0 * ops.ReduceMean()(positive_similar - ops.Log()(negative_similar))
        
        return loss