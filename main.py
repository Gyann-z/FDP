from torch.utils.data import Dataset, DataLoader
import torch
import clip
import os
import numpy as np
import random
from PIL import Image
import torch.nn as nn
from torch import optim
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from mmdet.core import PolygonMasks
from mmdet.core import BitmapMasks
import operator
from editdistance import eval
import torch.nn.functional as F
from fdp import CustomCLIP
from utils import generate_kernels,generate_effective_mask,bitmasks2tensor,balance_bce_loss

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

class image_caption_dataset(Dataset):
    def __init__(self, df, preprocess,input_resolution):
        self.images = df["image"]
        self.caption = df["caption"]
        self.polys = df['polys']
        self.polys_ignore = df['polys_ignore']
        self.preprocess = preprocess
        self.input_resolution=input_resolution

    def __len__(self):
        return len(self.caption)

    def __getitem__(self, idx):
        images = Image.open(self.images[idx])
        if 'gif' in self.images[idx] or 'png' in self.images[idx]:
            images = images.convert("RGB")
        img_shape = images.size
        w = img_shape[0]
        h = img_shape[1]

        images = self.preprocess(images)
        rx = self.input_resolution / w
        ry = self.input_resolution / h
        new_polys = []
        for poly in self.polys[idx]:
            new_polys.append([np.array([poly[0][0] * rx, poly[0][1] * ry, poly[0][2] * rx, poly[0][3] * ry,
                                        poly[0][4] * rx, poly[0][5] * ry, poly[0][6] * rx, poly[0][7] * ry])])

        gt_masks = PolygonMasks(new_polys, self.input_resolution, self.input_resolution)
        polygons = gt_masks.masks
        gt_shrink = generate_kernels(img_size=(self.input_resolution, self.input_resolution), text_polys=polygons, shrink_ratio=0.4,
                                     ignore_tags=None)

        new_polys_ignore = []
        for poly in self.polys_ignore[idx]:
            new_polys_ignore.append([np.array([poly[0][0] * rx, poly[0][1] * ry, poly[0][2] * rx, poly[0][3] * ry,
                                               poly[0][4] * rx, poly[0][5] * ry, poly[0][6] * rx, poly[0][7] * ry])])

        gt_masks_ignore = PolygonMasks(new_polys_ignore, self.input_resolution, self.input_resolution)
        polygons_ignore = gt_masks_ignore.masks
        gt_shrink_mask = generate_effective_mask((self.input_resolution, self.input_resolution), polygons_ignore)

        caption = self.caption[idx]
        return images, caption, gt_shrink, gt_shrink_mask

def load_data(img_path, gt_path, batch_size, preprocess,input_resolution):
    df = {'image': [], 'caption': [], 'polys': [], 'polys_ignore': []}
    dictionary = open("./dict/dictionary.txt").read().replace("\n\n", "\n").split("\n")
    img_list = os.listdir(img_path)
    gt_list = os.listdir(gt_path)

    for img in img_list:
        img_pth = os.path.join(img_path, img)
        gt = img.replace('jpg', 'txt').replace('png', 'txt').replace('gif', 'txt')
        polys = []
        polys_ignore = []
        if gt in gt_list:
            gt_pth = os.path.join(gt_path, gt)
            with open(gt_pth, 'r') as f:
                for line in f.readlines():
                    line_split = line.strip().split(',')
                    language = line_split[-2]
                    words = line_split[-1]
                    if language == 'Latin' and words != '' and '###' not in words:
                        polys.append(
                            [np.array([int(line_split[0]), int(line_split[1]), int(line_split[2]), int(line_split[3]),
                                       int(line_split[4]), int(line_split[5]), int(line_split[6]),
                                       int(line_split[7])])])
                    else:
                        polys_ignore.append(
                            [np.array([int(line_split[0]), int(line_split[1]), int(line_split[2]), int(line_split[3]),
                                       int(line_split[4]), int(line_split[5]), int(line_split[6]),
                                       int(line_split[7])])])

            with open(gt_pth, 'r') as f:
                for line in f.readlines():
                    line_split = line.strip().split(',')
                    language = line_split[-2]
                    words = line_split[-1]
                    if language == 'Latin' and words != '' and '###' not in words and len(words) > 2 and words.lower() in dictionary:
                        df['caption'].append(words.lower())
                        df['image'].append(img_pth)
                        df['polys'].append(polys)
                        df['polys_ignore'].append(polys_ignore)

    df_shuffle = list(zip(df['caption'], df['image'], df['polys'], df['polys_ignore']))
    print('caption num:', len(df['caption']))
    print('image num:', len(df['image']))
    print('polys num', len(df['polys']))
    random.shuffle(df_shuffle)
    df['caption'], df['image'], df['polys'], df['polys_ignore'] = zip(*df_shuffle)

    dataset = image_caption_dataset(df, preprocess,input_resolution)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    return train_dataloader

def train(epoch, batch_size, learning_rate, img_path, gt_path):
    CLIP_model, preprocess = clip.load('RN50', device=device, jit=False)
    input_resolution = 640
    preprocess = Compose([
        Resize(input_resolution, interpolation=BICUBIC),
        CenterCrop(input_resolution),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    CLIP_model = CLIP_model.float().eval()
    model = CustomCLIP(device, CLIP_model)

    for name, param in model.named_parameters():
        if "prompt_learner" not in name and "cross_attention_textquery" not in name and "new_positional_embedding" not in name\
                and "new_v_proj" not in name and "new_c_proj" not in name and "binarize" not in name:
            param.requires_grad_(False)
        # else:
        #     print(name)

    training_modules = [{"params": model.prompt_learner.parameters(), "lr": 2e-3},
                        {"params": model.cross_attention_textquery.parameters(), "lr": 2e-3},
                        {"params": model.new_positional_embedding, "lr": 1e-4},
                        {"params": model.new_v_proj.parameters(), "lr": 1e-4},
                        {"params": model.new_c_proj.parameters(), "lr": 1e-4},
                        {"params": model.binarize.parameters(), "lr": 2e-3}]
    # 加载数据集
    train_dataloader = load_data(img_path, gt_path, batch_size, preprocess, input_resolution)

    # 设置参数
    loss_img = nn.CrossEntropyLoss().to(device)
    loss_txt = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(training_modules, lr=learning_rate, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.001)

    dictionary = open("./dict/dictionary.txt").read().replace("\n\n", "\n").split("\n")
    from dict_trie import Trie
    trie = Trie(dictionary)

    for i in range(epoch):
        if i > 0 and i % 10 == 0:
            for p in optimizer.param_groups:
                p['lr'] *= 0.1

        for batch in train_dataloader:
            list_image, list_txt, gt_shrink, gt_shrink_mask = batch  # list_images is list of image in numpy array(np.uint8), or list of PIL images
            texts = list_txt
            images = list_image.to(device)
            gt_shrink = gt_shrink.to(device)
            gt_shrink_mask = gt_shrink_mask.to(device)

            all_candidates = []
            all_eval_distance = []
            for text in texts:
                eval_distance = []
                if len(text) > 5:
                    noise = 2
                else:
                    noise = 1
                candidates_list = list(trie.all_levenshtein(text, noise))
                candidates_list.append(text)
                candidates_list = list(set(candidates_list))
                candidates = {}
                for candidate in candidates_list:
                    candidates[candidate] = eval(text, candidate)
                candidates = sorted(candidates.items(), key=operator.itemgetter(1))
                dist_sharp = eval("###", text)
                while len(candidates) < 5:
                    candidates.append(("###", dist_sharp))
                candidates = candidates[:5]
                for can in candidates:
                    eval_distance.append(1 / (can[1] + 0.1))
                    all_candidates.append(can[0])
                all_eval_distance.append(eval_distance)
            all_eval_distance = torch.tensor(all_eval_distance).to(device)

            logits_per_image, logits_per_text, logits_candidates, score_map = model(texts, images, all_candidates)
            logits_candidates = nn.functional.softmax(logits_candidates, dim=1).log()
            all_eval_distance = nn.functional.softmax(all_eval_distance, dim=1)
            if device == "cpu":
                ground_truth = torch.arange(len(images)).long().to(device)
            else:
                ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

            feature_sz = tuple([gt_shrink.size(0), 1, 20, 20])
            downsample_ratio = 20 / input_resolution
            keys = ['gt_shrink', 'gt_shrink_mask']
            gt = {'gt_shrink': gt_shrink, 'gt_shrink_mask': gt_shrink_mask}

            for k in keys:
                gt[k] = [BitmapMasks([np.array(item.cpu())], input_resolution, input_resolution).rescale(downsample_ratio) for item in gt[k]]
                gt[k] = bitmasks2tensor(gt[k], feature_sz[2:])
                gt[k] = [item.to(score_map.device) for item in gt[k]]
            gt['gt_shrink'][0] = (gt['gt_shrink'][0] > 0).float()
            loss_prob = balance_bce_loss(score_map, gt['gt_shrink'][0], gt['gt_shrink_mask'][0])

            loss_ce = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            loss_kl_candidates = nn.KLDivLoss(reduction="batchmean")(logits_candidates, all_eval_distance)
            total_loss = loss_ce + loss_kl_candidates + loss_prob
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

        print('[%d] loss: %.3f' % (i + 1, total_loss), ' lr:', optimizer.state_dict()['param_groups'][0]['lr'])

        if i % 1 == 0:
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': total_loss, },
                       f'./save/fdp-' + str(i + 1) + 'th.pt')
            print(
                'save model to ./save/fdp-' + str(i + 1) + 'th.pt')


def main():
    epoch = 20
    batch_size = 48
    learning_rate = 2e-3
    img_path = './datasets/MLT/train_images/'
    gt_path = './datasets/MLT/train_gts/'
    train(epoch, batch_size, learning_rate, img_path, gt_path)


if __name__ == '__main__':
    main()