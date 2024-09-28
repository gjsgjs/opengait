import torch.utils.data as tordata
import cv2
import xarray as xr
from tqdm import tqdm
import random
import math
import torch
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os
from time import strftime, localtime


class DataSet(tordata.Dataset):
    def __init__(self, seq_dir, label, seq_type, view, cache, resolution):
        self.seq_dir = seq_dir
        self.view = view
        self.seq_type = seq_type
        self.label = label
        self.cache = cache
        self.resolution = int(resolution)
        self.cut_padding = int(float(resolution) / 64 * 10)
        self.data_size = len(self.label)
        self.data = [None] * self.data_size
        self.frame_set = [None] * self.data_size

        self.label_set = set(self.label)  # 去重 ，保存所有的人物标签
        self.seq_type_set = set(self.seq_type)  # 去重，保存最终的种类（bg-01。。。）
        self.view_set = set(self.view)
        _ = np.zeros((len(self.label_set),
                      len(self.seq_type_set),
                      len(self.view_set))).astype('int')
        _ -= 1
        self.index_dict = xr.DataArray(
            _,
            coords={'label': sorted(list(self.label_set)),
                    'seq_type': sorted(list(self.seq_type_set)),
                    'view': sorted(list(self.view_set))},
            dims=['label', 'seq_type', 'view'])

        for i in range(self.data_size):
            _label = self.label[i]
            _seq_type = self.seq_type[i]
            _view = self.view[i]
            self.index_dict.loc[_label, _seq_type, _view] = i

    def load_all_data(self):
        for i in tqdm(range(self.data_size)):
            self.load_data(i)

    def load_data(self, index):
        return self.__getitem__(index)

    def __loader__(self, path):
        return self.img2xarray(
            path)[:, :, self.cut_padding:-self.cut_padding].astype(
            'float32') / 255.0

    def img2xarray(self, flie_path):
        imgs = sorted(list(os.listdir(flie_path)))
        frame_list = [np.reshape(
            cv2.imread(osp.join(flie_path, _img_path)),
            [self.resolution, self.resolution, -1])[:, :, 0]
                      for _img_path in imgs
                      if osp.isfile(osp.join(flie_path, _img_path))]
        num_list = list(range(len(frame_list)))
        data_dict = xr.DataArray(
            frame_list,
            coords={'frame': num_list},
            dims=['frame', 'img_y', 'img_x'],
        )
        return data_dict

    def __getitem__(self, index):
        # pose sequence sampling
        if not self.cache:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
        elif self.data[index] is None:
            data = [self.__loader__(_path) for _path in self.seq_dir[index]]
            frame_set = [set(feature.coords['frame'].values.tolist()) for feature in data]
            frame_set = list(set.intersection(*frame_set))
            self.data[index] = data
            self.frame_set[index] = frame_set
        else:
            data = self.data[index]
            frame_set = self.frame_set[index]

        return data, frame_set, self.view[
            index], self.seq_type[index], self.label[index]

    def __len__(self):
        return len(self.label)


def load_data(dataset_path, resolution, dataset, cache=True):
    if cache == True:
        dataset_path = osp.join(dataset_path, 'train')
    else:
        dataset_path = osp.join(dataset_path, 'test')
    seq_dir = list()
    label = list()
    seq_type = list()
    view = list()

    for _label in sorted(list(os.listdir(dataset_path))):
        label_path = osp.join(dataset_path, _label)
        for _seq_type in sorted(list(os.listdir(label_path))):
            seq_type_path = osp.join(label_path, _seq_type)
            if cache == False:
                _view = '000'
                _seq_dir = osp.join(label_path, _seq_type)
                seqs = os.listdir(_seq_dir)
                if len(seqs) > 0:
                    seq_dir.append([_seq_dir])
                    label.append(_label)
                    seq_type.append(_seq_type)
                    view.append(_view)
            else:
                for _view in sorted(list(os.listdir(seq_type_path))):
                    _seq_dir = osp.join(seq_type_path, _view)
                    seqs = os.listdir(_seq_dir)
                    if len(seqs) > 0:
                        seq_dir.append([_seq_dir])
                        label.append(_label)
                        seq_type.append(_seq_type)
                        view.append(_view)

    if cache == False:
        data_source = DataSet(seq_dir, label, seq_type, view, cache, resolution)

    return data_source


def my_collate_fn(batch):
    sample_type = 'all'  # 全抽取
    frame_num = 30  # random时才有用
    # i为第几个 train=4*4 ,test=1
    # frame*64*44
    # batch[i][0][0] (101, 64, 44) 第i个视频帧数
    # batch[i][1] (101,) 第i个视频帧数
    # batch[i][2] view
    # batch[i][3] 视频序列号
    # batch[i][4] person
    batch_size = len(batch)
    feature_num = len(batch[0][0])
    seqs = [batch[i][0] for i in range(batch_size)]  # batch[0][0]
    frame_sets = [batch[i][1] for i in range(batch_size)]
    view = [batch[i][2] for i in range(batch_size)]
    seq_type = [batch[i][3] for i in range(batch_size)]
    label = [batch[i][4] for i in range(batch_size)]

    batch = [seqs, view, seq_type, label, None]

    # batch [5][i]

    def select_frame(index):
        sample = seqs[index]
        frame_set = frame_sets[index]
        if sample_type == 'random':
            frame_id_list = random.choices(frame_set, k=frame_num)
            _ = [feature.loc[frame_id_list].values for feature in sample]
        else:
            _ = [feature.values for feature in sample]
        return _

    seqs = list(map(select_frame, range(len(seqs))))  # select_frame -> (30,64,44)

    if sample_type == 'random':
        seqs = [np.asarray([seqs[i][j] for i in range(batch_size)]) for j in range(feature_num)]
    else:  # 全采样
        gpu_num = 1
        batch_per_gpu = math.ceil(batch_size / gpu_num)
        batch_frames = [[
            len(frame_sets[i])
            for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
            if i < batch_size
        ] for _ in range(gpu_num)]  # 全采样时每一个batch的每一个seq的帧数
        if len(batch_frames[-1]) != batch_per_gpu:
            for _ in range(batch_per_gpu - len(batch_frames[-1])):
                batch_frames[-1].append(0)
        max_sum_frame = np.max([np.sum(batch_frames[_]) for _ in range(gpu_num)])  # 每一个batch总帧数
        seqs = [[
            np.concatenate([
                seqs[i][j]
                for i in range(batch_per_gpu * _, batch_per_gpu * (_ + 1))
                if i < batch_size
            ], 0) for _ in range(gpu_num)]
            for j in range(feature_num)]
        seqs = [np.asarray([
            np.pad(seqs[j][_],
                   ((0, max_sum_frame - seqs[j][_].shape[0]), (0, 0), (0, 0)),
                   'constant',
                   constant_values=0)
            for _ in range(gpu_num)])
            for j in range(feature_num)]
        #
        batch[4] = np.asarray(batch_frames)

    batch[0] = seqs

    # batch[0][0] (1,1273,64,44) 一个batch所有帧 (30采样)加起来等于4*4*30
    # batch[1][i] view
    # batch[2][i] 视频动作序列号
    # batch[3][i] person
    # batch[4][i] 视频帧数 batch加起来(全采样)等于1273 (30采样)加起来等于4*4*30
    return batch





def cuda_dist(x, y, metric='euc'):
    x = torch.from_numpy(x)
    y = torch.from_numpy(y)
    if metric == 'cos':
        x = F.normalize(x, p=2, dim=1)  # [n c p]
        y = F.normalize(y, p=2, dim=1)  # [n c p]
    num_bin = x.size(2)  # 一个人一个视频多少个特征向量 (62)
    n_x = x.size(0)
    n_y = y.size(0)
    dist = torch.zeros(n_x, n_y)
    for i in range(num_bin):
        _x = x[:, :, i]  # _x.shape (n_x ,hidden_layer)
        _y = y[:, :, i]  # _y.shape (n_y ,hidden_layer)
        if metric == 'cos':
            dist += torch.matmul(_x, _y.transpose(0, 1))  # 余弦相似度,越近值越大
        else:
            _dist = torch.sum(_x ** 2, 1).unsqueeze(1) + torch.sum(_y ** 2, 1).unsqueeze(0) - 2 * torch.matmul(_x,
                                                                                                               _y.transpose(
                                                                                                                   0,
                                                                                                                   1))
            dist += torch.sqrt(F.relu(_dist))
    return 1 - dist / num_bin if metric == 'cos' else dist / num_bin

def print_log(message):
    print('[{0}] [INFO]: {1}'.format(strftime('%Y-%m-%d %H:%M:%S', localtime()), message))


def evaluation(data, dataset, metric='euc'):
    # 预测向量[all_nums, 128, 62]
    feature, _, seq_type, label = data
    label = np.array(label)
    seq_type = np.array(seq_type)
    probe_mask = (label == "probe")
    gallery_mask = (label != "probe")
    # 图库集（gallery）
    gallery_x = feature[gallery_mask, :]
    gallery_y = label[gallery_mask]
    # 探针集（probe）
    probe_x = feature[probe_mask, :]
    probe_y = seq_type[probe_mask]

    dist = cuda_dist(probe_x, gallery_x, metric)  # (x_n,y_n) distance

    idx = dist.sort(1)[1].numpy()  # (x_n,y_n) 表示原始矩阵中的元素排序后的位置
    save_path = osp.join(
        "output/result/result.csv")
    os.makedirs("output/result/", exist_ok=True)
    with open(save_path, "w") as f:
        f.write("videoID, label\n")
        for i in range(len(idx)):  # (x_n)
            f.write("{},{}\n".format(probe_y[i], gallery_y[idx[i, 0]]))  # 返回 x_n最近的y_n
        print_log("Your test result is saved to {}: {}".format(os.getcwd(), save_path))
