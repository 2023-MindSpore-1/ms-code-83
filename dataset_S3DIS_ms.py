import numpy as np
import mindspore as ms
from mindspore import dataset as ds
import time, pickle, glob, random
from os.path import join
from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP

class S3DIS:
    def __init__(self, test_area_idx, labeled_point):
        self.name = 'S3DIS'
        self.path = '/home/ubuntu/data/hdd1/wzh/S3DIS'
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter',
                               13: 'unlabel'
                               }
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([13])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.s_indx = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size,  labeled_point)
        self.init_input_pipeline()

    def load_sub_sampled_clouds(self, sub_grid_size, labeled_point):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):

            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']
            sub_xyz = np.vstack((data['x'], data['y'], data['z'])).T
            all_select_label_indx = []
            if cloud_split == 'training' :
                ''' ***************** '''
                all_select_label_indx = []
                for i in range(self.num_classes):
                    ind_class = np.where(sub_labels == i)[0]
                    num_classs = len(ind_class)
                    if num_classs > 0:
                        if '%' in labeled_point:
                            r = float(labeled_point[:-1]) / 100
                            num_selected = max(int(num_classs * r), 1)
                        else:
                            num_selected = int(labeled_point)

                        label_indx = list(range(num_classs))
                        random.shuffle(label_indx)
                        noselect_labels_indx = label_indx[num_selected:]
                        select_labels_indx = label_indx[:num_selected]
                        ind_class_noselect = ind_class[noselect_labels_indx]
                        ind_class_select = ind_class[select_labels_indx]
                        all_select_label_indx.append(ind_class_select[0])
                        sub_labels[ind_class_noselect] = 13
                ''' ***************** '''
            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]
            if cloud_split == 'training':
                self.s_indx[cloud_split] += [all_select_label_indx]  # training only]:

            size = sub_colors.shape[0] * 4 * 10
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                if split=='training':
                    s_indx = self.s_indx[split][cloud_idx]#training only
                    # Shuffle index
                    queried_idx = np.concatenate([np.array(s_indx), queried_idx],0)[:cfg.num_points]#training only

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = [ms.float32, ms.float32, ms.int32, ms.int32, ms.int32]
        gen_names = ["queried_pc_xyz", "queried_pc_colors", "queried_pc_labels", "queried_idx", "cloud_idx"]
        return gen_func, gen_types, gen_names

    @staticmethod
    def get_ms_mapping2():
        # Collect flat inputs
        def ms_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = np.concatenate([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = DP.knn_search(batch_xyz, batch_xyz, cfg.k_n)
                sub_points = batch_xyz[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :batch_xyz.shape[1] // cfg.sub_sampling_ratio[i], :]
                up_i = DP.knn_search(sub_points, batch_xyz, 1)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return tuple(input_list)

        columns = ["input_points_%d"%i for i in range(cfg.num_layers)] + ["input_neighbors_%d"%i for i in range(cfg.num_layers)] \
                + ["input_pools_%d"%i for i in range(cfg.num_layers)] + ["input_up_samples_%d"%i for i in range(cfg.num_layers)] \
                + ["batch_features", "batch_labels", "batch_pc_idx", "batch_cloud_idx"]
        return ms_map, columns

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_names = self.get_batch_gen('training')
        self.train_data = ds.GeneratorDataset(source=gen_function, column_names=gen_names, column_types=gen_types)
        gen_function_val, gen_types, gen_names = self.get_batch_gen('validation')
        self.val_data = ds.GeneratorDataset(source=gen_function_val, column_names=gen_names, column_types=gen_types)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func, columns = self.get_ms_mapping2()

        self.batch_train_data = self.batch_train_data.map(operations=map_func, 
            input_columns=gen_names, output_columns=columns, column_order=columns)
        self.batch_val_data = self.batch_val_data.map(operations=map_func, 
            input_columns=gen_names, output_columns=columns, column_order=columns)
