from pickletools import optimize
import numpy as np
import mindspore as ms
from mindspore import numpy as msnp
from mindspore import nn, ops, Tensor
import os, argparse, shutil
import Loss_ms
from eval import evaluate
from HybridCR_ms import HybridCR
from dataset_S3DIS_ms import S3DIS
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP

def create_log_dir(args):
    '''CREATE DIR'''
    import datetime,sys
    from pathlib import Path

    if args.mode == 'train':
        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        experiment_dir = Path('./experiment/')
        experiment_dir.mkdir(exist_ok=True)
        experiment_dir = experiment_dir.joinpath('S3DIS')
        experiment_dir.mkdir(exist_ok=True)
        if '%' in args.labeled_point:
            n = args.labeled_point[:-1] + '_percent_'
        else:
            n = args.labeled_point + '_points_'

        experiment_dir = experiment_dir.joinpath(n) 
        experiment_dir.mkdir(exist_ok=True)
        if args.log_dir is None:
            experiment_dir = experiment_dir.joinpath(timestr + '_area_' + str(args.test_area))
        else:
            experiment_dir = experiment_dir.joinpath(args.log_dir)
        experiment_dir.mkdir(exist_ok=True)
        checkpoints_dir = experiment_dir.joinpath('checkpoints/')
        checkpoints_dir.mkdir(exist_ok=True)
        tensorboard_log_dir = experiment_dir.joinpath('tensorboard/')
        tensorboard_log_dir.mkdir(exist_ok=True)
        shutil.copy('helper_tool.py', str(experiment_dir))
        f = sys.argv[0]
        shutil.copy(f, str(experiment_dir))
        try:
            shutil.copy(args.model_name, str(experiment_dir))
        except:
            print('coppy file error')
            1/0
    elif args.mode == 'test':
        model_path = args.model_path
        checkpoints_dir = model_path.split('snapshots')[0]
        log_dir = os.path.join(model_path.split('snapshots')[0],'logs')
        experiment_dir = model_path.split('checkpoints')[0]
    return str(experiment_dir), str(checkpoints_dir), str(tensorboard_log_dir)

class WithLoss(nn.Cell):
    def __init__(self, network, loss, embedding):
        super(WithLoss, self).__init__()
        self.network = network
        self.loss = loss
        self.embedding = embedding
    
    def construct(self, data, labels, neigh_idx, class_weights, num_classes, ignored_label_inds):
        logits, embedding = self.network(data)
        labels = ops.Concat(axis=-1)([labels, labels])
        valid_idx, class_embedding = self.embedding(logits, labels, embedding, num_classes, ignored_label_inds)
        loss = self.loss(logits, labels, embedding, valid_idx, class_embedding, neigh_idx, class_weights)
        return loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MSmode', type=str, default='PYNATIVE_MODE', choices=['GRAPH_MODE', 'PYNATIVE_MODE'])
    parser.add_argument('--gpu', type=int, default=1, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--labeled_point', type=str, default='1%', choices=['1', '1%'])
    parser.add_argument('--model_name', type=str, default='HybridCR_ms.py')
    parser.add_argument('--log_dir', type=str, default= 'HybridCR_Area-5')
    parser.add_argument('--knn', type=int, default=16)
    FLAGS = parser.parse_args()

    GPU_ID = FLAGS.gpu
    Mode = FLAGS.MSmode
    test_area = FLAGS.test_area
    # ms.set_context(mode=ms.GRAPH_MODE, device_id=GPU_ID)
    ms.set_context(mode=ms.PYNATIVE_MODE, device_id=GPU_ID)

    s3dis = S3DIS(test_area, FLAGS.labeled_point)
    dataset_train = s3dis.batch_train_data
    dataset_val = s3dis.batch_val_data
    class_weights = Tensor(DP.get_class_weights("S3DIS"), ms.float32)

    network = HybridCR(cfg)
    loss = Loss_ms.Loss(cfg.batch_size, cfg.num_classes, cfg.num_points)
    net_with_loss = WithLoss(network, loss, Loss_ms.global_embedding)
    
    lr = nn.exponential_decay_lr(cfg.learning_rate, cfg.lr_decays, cfg.train_steps * cfg.max_epoch, cfg.train_steps, 1)
    optimizer = nn.Adam(network.trainable_params(), learning_rate=lr)
    train_net = nn.TrainOneStepCell(net_with_loss, optimizer)

    num_layers = cfg.num_layers
    for i in range(cfg.max_epoch):
        print("****EPOCH %d****"%(i))
        for step_id, item in enumerate(dataset_train.create_tuple_iterator()):
            data = dict()
            data['xyz'] = item[: num_layers]
            data['neigh_idx'] = item[num_layers : 2 * num_layers]
            data['sub_idx'] = item[2 * num_layers : 3 * num_layers]
            data['interp_idx'] = item[3 * num_layers:4 * num_layers]
            data['features'] = item[4 * num_layers]
            data['labels'] = item[4 * num_layers + 1]
            data['input_inds'] = item[4 * num_layers + 2]
            data['cloud_inds'] = item[4 * num_layers + 3]
            
            
            # print("start")
            # labels = ops.Concat(axis=-1)([data['labels'], data['labels']])
            # logits, embedding = network(item)
            # # print("logits.shape: ", logits.shape)
            # # print("embedding.shape: ", embedding.shape)
            # valid_idx, class_embedding = Loss_ms.global_embedding(logits, labels, embedding, cfg.num_classes, cfg.ignored_label_inds)
            # print(len(valid_idx))
            # loss = Loss_ms.Loss(cfg.batch_size, cfg.num_classes, cfg.num_points)
            # out = loss(logits, labels, embedding, valid_idx, class_embedding, data['neigh_idx'][0], class_weights)
            # print("loss: ", out)
            # print("end")


            network.set_train(True)
            train_net(item, item[4 * num_layers + 1], item[num_layers], class_weights, cfg.num_classes, cfg.ignored_label_inds)
            loss = net_with_loss(item, item[4 * num_layers + 1], item[num_layers], class_weights, cfg.num_classes, cfg.ignored_label_inds)
            if step_id % 50 == 0: print("step", step_id, loss)

        #     break
        # break
        network.set_train(False)
        m_iou = evaluate(dataset_val, network, cfg)
    
    # dataset.experiment_dir, dataset.checkpoints_dir, dataset.tensorboard_log_dir = create_log_dir(FLAGS)

    # if FLAGS.knn is not None:
    #     cfg.k_n = FLAGS.knn

    # if Mode == 'train':
    #     model = Network(dataset, cfg)
    #     model.train(dataset)