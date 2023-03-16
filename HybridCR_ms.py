import numpy as np
import mindspore as ms
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import initializer, Normal
from helper_tool import ConfigS3DIS as cfg


class SharedMLP(nn.Cell):           # d_in -> d_out
    def __init__(self, d_in, d_out, kernel_size=1, stride=1, pad_mode='valid',
                 bn=False, activation_fn=None, transpose=False):
        super(SharedMLP, self).__init__()
        conv_fn = nn.Conv2dTranspose if transpose else nn.Conv2d
        self.conv = conv_fn(d_in, d_out, kernel_size, stride, pad_mode,
                            has_bias=True, weight_init="TruncatedNormal", bias_init="zeros")
        self.batch_norm = nn.BatchNorm2d(d_out, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def construct(self, inputs):
        outputs = self.conv(inputs)
        if self.batch_norm:
            outputs = self.batch_norm(outputs)
        if self.activation_fn:
            outputs = self.activation_fn(outputs)
        return outputs

class Gather(nn.Cell):
    def __init__(self):
        super(Gather, self).__init__()

    def construct(self, pc, neigh_idx):
        pc = ops.Tile()(pc.expand_dims(-1), (1, 1, 1, neigh_idx.shape[-1]))
        neigh_idx = ops.Tile()(neigh_idx.expand_dims(1), (1, pc.shape[1], 1, 1))
        features = ops.GatherD()(pc, 2, neigh_idx)
        return features

class LocalSpatialEncoding(nn.Cell):
    def __init__(self):
        super(LocalSpatialEncoding, self).__init__()

    def construct(self, xyz, neigh_idx):
        xyz = xyz.transpose((0, 2, 1))
        neighbor_xyz = Gather()(xyz, neigh_idx)
        xyz_tile = ops.Tile()(xyz.expand_dims(-1), (1, 1, 1, neigh_idx.shape[-1]))
        relative_xyz = xyz_tile - neighbor_xyz
        relative_dis = ops.Sqrt()(ops.ReduceSum(keep_dims=True)(ops.Square()(relative_xyz), 1))
        relative_feature = ops.Concat(axis=1)([relative_dis, relative_xyz, xyz_tile, neighbor_xyz])
        return relative_feature

class AttentivePooling(nn.Cell):
    def __init__(self, d_in, d_out):
        super(AttentivePooling, self).__init__()
        self.fc = nn.Dense(d_in, d_in, has_bias=False)
        self.softmax = nn.Softmax(axis=-2)
        self.mlp = SharedMLP(d_in, d_out, bn=True, activation_fn=nn.LeakyReLU())

    def construct(self, feature_set):
        att_activation = self.fc(feature_set.transpose((0, 2, 3, 1)))
        att_scores = self.softmax(att_activation).transpose((0, 3, 1, 2))
        f_agg = feature_set * att_scores
        f_agg = ops.ReduceSum(keep_dims=True)(f_agg, -1)
        f_agg = self.mlp(f_agg)
        return f_agg

class DilatedResBlock(nn.Cell):     # d_in -> d_out * 2
    def __init__(self, d_in, d_out):
        super(DilatedResBlock, self).__init__()
        self.mlp1 = SharedMLP(d_in, d_out // 2, bn=True, activation_fn=nn.LeakyReLU())
        self.mlp2 = SharedMLP(d_out, d_out * 2, bn=True)
        self.shortcut = SharedMLP(d_in, d_out * 2, bn=True)

        self.mlp3 = SharedMLP(10, d_in, bn=True, activation_fn=nn.LeakyReLU())
        self.mlp4 = SharedMLP(d_in, d_out // 2, bn=True, activation_fn=nn.LeakyReLU())

        self.pool1 = AttentivePooling(d_in + d_out // 2, d_out // 2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.leaky_relu = nn.LeakyReLU()

    def construct(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)
        f_xyz = LocalSpatialEncoding()(xyz, neigh_idx)

        f_xyz = self.mlp3(f_xyz)
        f_neighbours = Gather()(f_pc.squeeze(-1), neigh_idx)
        f_pc_agg = self.pool1(ops.Concat(axis=1)([f_neighbours, f_xyz]))

        f_xyz = self.mlp4(f_xyz)
        f_neighbours = Gather()(f_pc_agg.squeeze(-1), neigh_idx)
        f_pc_agg = self.pool2(ops.Concat(axis=1)([f_neighbours, f_xyz]))

        f_pc = self.mlp2(f_pc_agg)
        shortcut = self.shortcut(feature)
        return self.leaky_relu(f_pc + shortcut)

class Augmentor(nn.Cell):
    def __init__(self, num_classes, batch_size, num_points):
        super(Augmentor, self).__init__()
        self.noise1 = initializer(Normal(sigma=0.8, mean=0.0), [batch_size, num_classes, num_points, 1], ms.float32)
        self.noise2 = initializer(Normal(sigma=0.8, mean=0.0), [batch_size, num_classes, num_points, 1], ms.float32)

        self.fc0 = nn.Dense(6, 6)
        self.bn = nn.BatchNorm2d(6, 1e-6, 0.99)
        self.leaky_relu = nn.LeakyReLU()

        self.fc1 = SharedMLP(6, 64, bn=True, activation_fn=nn.LeakyReLU())
        self.fc2 = SharedMLP(64, 128, bn=True, activation_fn=nn.LeakyReLU())
        self.fc3 = SharedMLP(128, 1024, bn=True, activation_fn=nn.LeakyReLU())
        self.fc4 = SharedMLP(1024, 512, bn=True, activation_fn=nn.LeakyReLU())
        self.fc_f = SharedMLP(512, num_classes)

        self.fc_m = SharedMLP(num_classes * 2, 3)
        self.fc_d = SharedMLP(num_classes * 2, 3)
    
    def construct(self, inputs):
        feature = inputs
        points = inputs[:, :, 0:3]
        colors = inputs[:, :, 3:6]

        feature = self.fc0(feature).transpose((0, 2, 1)).expand_dims(-1)      # (batch_size, channels, num_points, 1)
        feature = self.bn(feature)
        feature = self.leaky_relu(feature)

        f_layer_fc1 = self.fc1(feature)
        f_layer_fc2 = self.fc2(f_layer_fc1)
        f_layer_fc3 = self.fc3(f_layer_fc2)
        f_layer_fc4 = self.fc4(f_layer_fc3)
        f_layer_f = self.fc_f(f_layer_fc4)

        feature_M = self.fc_m(ops.Concat(axis=1)([f_layer_f, self.noise1]))
        feature_M = ops.Tile()(feature_M, (1, 1, 1, 3))
        points = ops.Tile()(points.expand_dims(-2), (1, 1, 3, 1))
        feature_M = ops.matmul(points, feature_M.transpose((0, 2, 1, 3)))
        feature_M = ops.ReduceSum()(feature_M, -2)

        feature_D = self.fc_d(ops.Concat(axis=1)([f_layer_f, self.noise2]))
        aug_feature = ops.Add()(feature_M.expand_dims(2), feature_D.transpose((0, 2, 3, 1)))
        aug_feature = aug_feature.squeeze(2)
        aug_feature = ops.Concat(axis=-1)([aug_feature, colors])

        return aug_feature

class Srcontext(nn.Cell):
    def __init__(self, d_in):
        super(Srcontext, self).__init__()
        self.mlp = SharedMLP(d_in * 2, d_in)
        self.norm = nn.Norm(axis=1, keep_dims=True)

    def construct(self, feature, neigh_idx):
        feature_sq = feature.squeeze(-1)
        if self.training:
            neigh_idx = ops.Concat(axis=0)([neigh_idx, neigh_idx])
        neighbor_feature = Gather()(feature_sq, neigh_idx)
        feature_tile = ops.Tile()(feature_sq.expand_dims(-1), (1, 1, 1, neigh_idx.shape[-1]))
        relative_feature = feature_tile - neighbor_feature
        edge_feature = ops.Concat(axis=1)([relative_feature, feature_tile])
        feature_rs1 = self.mlp(edge_feature)
        feature_rs1 = ops.ReduceMax(keep_dims=False)(feature_rs1, -1)
        rs_map_s1 = self.norm(feature_rs1)
        rs_mapf1 = feature_rs1 / rs_map_s1
        fea_out = ops.Concat(axis=1)([feature_sq, feature_rs1])
        fea_out = fea_out.expand_dims(-1)
        return fea_out, rs_mapf1

class HybridCR(nn.Cell):
    def __init__(self, config):
        super(HybridCR, self).__init__()
        self.d_out = config.d_out
        self.num_layers = config.num_layers

        self.data_aug = Augmentor(config.num_classes, config.batch_size, config.num_points)
        self.fc0 = nn.Dense(6, 8)
        self.bn1 = nn.BatchNorm2d(8, 1e-6, 0.99)
        self.leaky_relu = nn.LeakyReLU()

        # Encoding Layers
        # 8 -> 32 -> 128 -> 256 -> 512 -> 1024
        self.encoder = nn.CellList([DilatedResBlock(8, self.d_out[0])])
        for i in range(1, self.num_layers):
            self.encoder.append(DilatedResBlock(self.d_out[i-1] * 2, self.d_out[i]))
        
        self.mlp = SharedMLP(self.d_out[-1] * 2, self.d_out[-1] * 2, activation_fn=nn.LeakyReLU())

        # Decoding Layers
        # 1024 + 512 -> 512 + 256 -> 256 + 128 -> 128 + 32 -> 32 + 32 -> 32
        self.decoder = nn.CellList([])
        for j in range(1, self.num_layers):
            self.decoder.append(SharedMLP((self.d_out[-j] + self.d_out[-j-1]) * 2, self.d_out[-j-1] * 2, transpose=True, bn=True, activation_fn=nn.LeakyReLU()))
        self.decoder.append(SharedMLP(self.d_out[0] * 4, self.d_out[0] * 2, transpose=True, bn=True, activation_fn=nn.LeakyReLU()))

        # Final Semantic Prediction
        self.fc1 = SharedMLP(self.d_out[0] * 2, 32, bn=True, activation_fn=nn.LeakyReLU())
        self.edgeconv = Srcontext(32)
        self.fc2 = SharedMLP(64, 32, bn=True, activation_fn=nn.LeakyReLU())
        self.dp = nn.Dropout()
        self.fc3 = SharedMLP(32, config.num_classes)

    def construct(self, dataset):
        data = dict()
        data['xyz'] = dataset[: self.num_layers]                                # Float32   (batch_size, num_points, 3)
        data['neigh_idx'] = dataset[self.num_layers : 2 * self.num_layers]      # Int32     (batch_size, num_points, k_n)
        data['sub_idx'] = dataset[2 * self.num_layers : 3 * self.num_layers]    # Int32     (batch_size, num_points / sub_sampling_ratio, k_n)
        data['interp_idx'] = dataset[3 * self.num_layers : 4 * self.num_layers] # Int32     (batch_size, num_points, 1)
        data['features'] = dataset[4 * self.num_layers]                         # Float32   (batch_size, num_points, 6)
        data['labels'] = dataset[4 * self.num_layers + 1]                       # Int32     (batch_size, num_points)
        data['input_inds'] = dataset[4 * self.num_layers + 2]                   # Int32     (batch_size, num_points)
        data['cloud_inds'] = dataset[4 * self.num_layers + 3]                   # Int32     (batch_size, 1)
        
        feature = data['features']
        if self.training:
            feature = ops.Concat(axis=0)([feature, self.data_aug(feature)])
        feature = self.fc0(feature).transpose((0, 2, 1)).expand_dims(-1)        # (batch_size, channels, num_points, 1)
        feature = self.leaky_relu(self.bn1(feature))

        f_encoder_list = []
        for i in range(self.num_layers):
            xyz = data['xyz'][i]
            neigh_idx = data['neigh_idx'][i]
            sub_idx = data['sub_idx'][i]
            if self.training:
                xyz = ops.Concat(axis=0)([xyz, xyz])
                neigh_idx = ops.Concat(axis=0)([neigh_idx, neigh_idx])
                sub_idx = ops.Concat(axis=0)([sub_idx, sub_idx])
            f_encoder_i = self.encoder[i](feature, xyz, neigh_idx)
            f_sampled_i = Gather()(f_encoder_i.squeeze(-1), sub_idx)
            f_sampled_i = ops.ReduceMax(keep_dims=True)(f_sampled_i, -1)
            feature = f_sampled_i
            if i == 0: f_encoder_list.append(f_encoder_i.copy())
            f_encoder_list.append(f_sampled_i.copy())

        feature = self.mlp(f_encoder_list[-1])

        f_decoder_list = []
        for j in range(self.num_layers):
            interp_idx = data['interp_idx'][-j-1]
            if self.training:
                interp_idx = ops.Concat(axis=0)([interp_idx, interp_idx])
            f_interp_i = Gather()(feature.squeeze(-1), interp_idx)
            f_decoder_i = self.decoder[j](ops.Concat(axis=1)([f_encoder_list[-j-2], f_interp_i]))
            feature = f_decoder_i
            f_decoder_list.append(f_decoder_i)

        f_layer_fc1 = self.fc1(f_decoder_list[-1])
        f_layer_fc1, rs_mapf1 = self.edgeconv(f_layer_fc1, data['neigh_idx'][0])
        f_layer_fc2 = self.fc2(f_layer_fc1)
        f_layer_drop = self.dp(f_layer_fc2)
        f_layer_fc3 = self.fc3(f_layer_drop)
        f_out = f_layer_fc3.squeeze(-1)

        return f_out.transpose((0, 2, 1)), rs_mapf1.transpose((0, 2, 1))


if __name__ == '__main__':
    model = HybridCR(cfg)
    print("parameters_dict of model:")
    for m in model.get_parameters():
        print(m)