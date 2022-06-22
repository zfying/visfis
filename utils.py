from __future__ import print_function

import errno
import os
from PIL import Image
import torch
import torch.nn as nn
import os
import json
import pickle as  cPickle
import numpy as np
import utils
import h5py
#from pycocotools.coco import COCO
from scipy.sparse import coo_matrix
from sklearn.metrics.pairwise import cosine_similarity

import pdb

EPS = 1e-7


def get_spatial_length(spatial_feature_type, spatial_feature_length):
    if spatial_feature_type == 'none':
        return 0
    elif spatial_feature_type == 'mesh':
        return 2 * spatial_feature_length * spatial_feature_length
    elif spatial_feature_type == 'linear':
        return 2 * spatial_feature_length
    elif spatial_feature_type == 'simple':
        return 4
    else:
        return 0
    
def get_oracle_length(oracle_type, oracle_embed_size):
    if oracle_type == 'none':
        return 0
    elif oracle_type == 'simple':
        return 1
    elif oracle_type == 'wordvec':
        return oracle_embed_size
    else:
        raise ValueError("Unknown oracle type", oracle_type)

def get_linear_features(curr_spatial_features, num_objects, spatial_feature_length):
    linear_features_x, linear_features_y = [], []
    for obj_ix in range(num_objects):
        x_start, x_end = curr_spatial_features[obj_ix][0], curr_spatial_features[obj_ix][2]
        y_start, y_end = curr_spatial_features[obj_ix][1], curr_spatial_features[obj_ix][3]
        curr_feats_x = np.linspace(x_start, x_end, num=spatial_feature_length)
        curr_feats_y = np.linspace(y_start, y_end, num=spatial_feature_length)
        linear_features_x.append(curr_feats_x.tolist())
        linear_features_y.append(curr_feats_y.tolist())

    linear_features_x, linear_features_y = np.array(linear_features_x), np.array(linear_features_y)
    return linear_features_x, linear_features_y

def normalize_features(curr_image_features):
    # return curr_image_features
    norm = np.linalg.norm(curr_image_features, axis=1)
    denom = np.repeat(norm, curr_image_features.shape[1]).reshape(
        (curr_image_features.shape[0], curr_image_features.shape[1]))
    curr_image_features = np.divide(curr_image_features, denom)
    return curr_image_features

def adding_spatials(opt, curr_image_features, curr_spatial_features, spatial_feature_type,
                           spatial_feature_length, num_objects):

    if spatial_feature_type == 'none':
        curr_entry = np.array(curr_image_features)
    elif spatial_feature_type == 'simple':
        # curr_spatial_features = normalize_features(curr_spatial_features)
        if opt.dataset != 'hatcp':
            curr_entry = np.concatenate((curr_image_features, curr_spatial_features[:, [0, 1, 4, 5]]), axis=1)
        else: # for weird vqa-hat
            # tmp = curr_spatial_features[:, [0, 1, 2, 3]]
            # tmp[:, [0,2]] /= curr_spatial_features[:,[6]]
            # tmp[:, [1,3]] /= curr_spatial_features[:,[5]]
            # if np.isnan(tmp).sum()!=0:
            #     pdb.set_trace()
            tmp = np.zeros((curr_spatial_features.shape[0], 4))
            curr_entry = np.concatenate((curr_image_features, tmp), axis=1)
    elif spatial_feature_type == 'linear':
        linear_features_x, linear_features_y = get_linear_features(curr_spatial_features, num_objects, spatial_feature_length)
        linear_features_x = normalize_features(linear_features_x)
        linear_features_y = normalize_features(linear_features_y)
        curr_entry = np.concatenate(
            (normalize_features(curr_image_features), linear_features_x, linear_features_y), axis=1)
    elif spatial_feature_type == 'mesh':
        linear_features_x, linear_features_y = get_linear_features(curr_spatial_features, num_objects, spatial_feature_length)
        meshes = []
        for obj_ix in range(num_objects):
            curr_mesh = np.array(np.meshgrid(linear_features_x[obj_ix], linear_features_y[obj_ix])).flatten()
            meshes.append(curr_mesh)
        curr_entry = np.concatenate((normalize_features(curr_image_features), meshes), axis=1)
    else:
        assert ValueError(f'Unsupported spatial_feature_type {spatial_feature_type}')

    return curr_entry

def adding_oracles(visual_features, hint_scores, oracle_type, oracle_embed=None):

    if oracle_type == 'none':
        curr_entry = np.array(visual_features)
    elif oracle_type == 'simple':
        curr_entry = np.concatenate((visual_features, hint_scores), axis=1)
    elif oracle_type == 'wordvec':
        assert(oracle_embed is not None)
        # change from np to torch
        visual_features = torch.from_numpy(visual_features)
        # apply mask
        oracle_mask = (hint_scores.squeeze() > 0.85).long()
        oracle_embedding = oracle_embed(oracle_mask)
        curr_entry =torch.cat([visual_features, oracle_embedding.float()], dim=1).detach().numpy()
    else:
        raise ValueError(f'unsupported oracle_type {oracle_type}')
        
    return curr_entry

def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def assert_array_eq(real, expected):
    assert (np.abs(real-expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def weights_init(m):
    """custom weights initialization."""
    cname = m.__class__
    if cname == nn.Linear or cname == nn.Conv2d or cname == nn.ConvTranspose2d:
        m.weight.data.normal_(0.0, 0.02)
    elif cname == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    else:
        print('%s is not initialized.' % cname)


def init_net(net, net_file):
    if net_file:
        net.load_state_dict(torch.load(net_file))
    else:
        net.apply(weights_init)


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Logger(object):
    def __init__(self, output_name):
        dirname = os.path.dirname(output_name)
        if not os.path.exists(dirname):
            os.mkdir(dirname)

        self.log_file = open(output_name, 'w')
        self.infos = {}

    def append(self, key, val):
        vals = self.infos.setdefault(key, [])
        vals.append(val)

    def log(self, extra_msg=''):
        msgs = [extra_msg]
        for key, vals in self.infos.iteritems():
            msgs.append('%s %.6f' % (key, np.mean(vals)))
        msg = '\n'.join(msgs)
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        self.infos = {}
        return msg

    def write(self, msg):
        self.log_file.write(msg + '\n')
        self.log_file.flush()
        print(msg)




def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GradMulConst(torch.autograd.Function):
    """
    This layer is used to create an adversarial loss.
    """
    @staticmethod
    def forward(ctx, x, const):
        ctx.const = const
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.const, None

def grad_mul_const(x, const):
    return GradMulConst.apply(x, const)