import torch
import cv2
import sys
from shapely.geometry import Polygon as plg
import pyclipper
import numpy as np
import clip
import torch.nn.functional as F
from sklearn.cluster import KMeans
device = "cuda" if torch.cuda.is_available() else "cpu"

def Kmeans_cluster():
    clip_model, preprocess = clip.load('RN50', device)
    queries = []
    with open('./dict/queries.txt', 'r') as f:
        for line in f:
            queries.append(line.replace('\n', ''))

    text_inputs = torch.cat([clip.tokenize(p) for p in queries]).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)

    num_clusters = 2
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(text_features.cpu().numpy())
    labels = kmeans.labels_

    for i in range(num_clusters):
        cluster_words = [queries[j] for j in range(len(queries)) if labels[j] == i]
        if 'the' in cluster_words:
            function_cluster = i
        if 'office' in cluster_words:
            content_cluster = i
    assert function_cluster != content_cluster
    # print('function_cluster', function_cluster)
    # print('content_cluster', content_cluster)

    return kmeans,function_cluster,content_cluster

def generate_kernels(img_size,
                     text_polys,
                     shrink_ratio,
                     max_shrink=sys.maxsize,
                     ignore_tags=None):
    assert isinstance(img_size, tuple)
    assert isinstance(shrink_ratio, float)

    h, w = img_size
    text_kernel = np.zeros((h, w), dtype=np.float32)

    for text_ind, poly in enumerate(text_polys):
        instance = poly[0].reshape(-1, 2).astype(np.int32)
        area = plg(instance).area
        peri = cv2.arcLength(instance, True)
        distance = min(
            int(area * (1 - shrink_ratio * shrink_ratio) / (peri + 0.001) +
                0.5), max_shrink)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(instance, pyclipper.JT_ROUND,
                    pyclipper.ET_CLOSEDPOLYGON)
        shrunk = np.array(pco.Execute(-distance))

        if len(shrunk) == 0 or shrunk.size == 0:
            if ignore_tags is not None:
                ignore_tags[text_ind] = True
            continue
        try:
            shrunk = np.array(shrunk[0]).reshape(-1, 2)

        except Exception as e:
            print_log(f'{shrunk} with error {e}')
            if ignore_tags is not None:
                ignore_tags[text_ind] = True
            continue
        cv2.fillPoly(text_kernel, [shrunk.astype(np.int32)], text_ind + 1)
    return text_kernel


def generate_effective_mask(mask_size: tuple, polygons_ignore):
    mask = np.ones(mask_size, dtype=np.uint8)

    for poly in polygons_ignore:
        instance = poly[0].reshape(-1, 2).astype(np.int32).reshape(1, -1, 2)
        cv2.fillPoly(mask, instance, 0)

    return mask

def bitmasks2tensor(bitmasks, target_sz):
    assert isinstance(bitmasks, list)
    assert isinstance(target_sz, tuple)

    batch_size = len(bitmasks)
    num_levels = len(bitmasks[0])

    result_tensors = []
    for level_inx in range(num_levels):
        kernel = []
        for batch_inx in range(batch_size):
            mask = torch.from_numpy(bitmasks[batch_inx].masks[level_inx])
            mask_sz = mask.shape
            pad = [
                0, target_sz[1] - mask_sz[1], 0, target_sz[0] - mask_sz[0]
            ]
            mask = F.pad(mask, pad, mode='constant', value=0)
            kernel.append(mask)
        kernel = torch.stack(kernel)
        result_tensors.append(kernel)

    return result_tensors

def balance_bce_loss(pred, gt, mask):
    positive = (gt * mask)
    negative = ((1 - gt) * mask)
    positive_count = int(positive.float().sum())
    negative_count = min(
        int(negative.float().sum()),
        int(positive_count * 3.0))

    assert gt.max() <= 1 and gt.min() >= 0
    loss = F.binary_cross_entropy_with_logits(pred, gt, reduction='none')
    positive_loss = loss * positive.float()
    negative_loss = loss * negative.float()

    negative_loss, _ = torch.topk(negative_loss.view(-1), negative_count)

    eps = 1e-6
    balance_loss = (positive_loss.sum() + negative_loss.sum()) / (
            positive_count + negative_count + eps)

    return balance_loss