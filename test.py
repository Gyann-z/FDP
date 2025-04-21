import glob
import scipy.io as io
from PIL import Image
from sklearn.metrics import average_precision_score
import clip
import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from fdp import CustomCLIP
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def test(model_path,gt_data_path,inp_path):
    CLIP_model, preprocess = clip.load('RN50', device=device)
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
    model.eval()

    model_load = torch.load(model_path)
    state_dict = model_load['model_state_dict']
    model.load_state_dict(state_dict)

    gt_data = io.matlab.loadmat(gt_data_path)
    str_queries = []
    for i in range(gt_data['data'].shape[1]):
        str_queries.append(str(gt_data['data'][0, i][0][0][0][0]).lower())
    print(str_queries)

    for i in range(len(str_queries)):
        query = str_queries[i]
        rel = []
        for j in range(len(gt_data['data'][0, i][1])):
            rel.append(str(gt_data['data'][0, i][1][j][0][0]).lower())
        # print("{}: {}\n".format(query, rel))

    all_inps = glob.glob(inp_path + '*.jpg')
    if len(all_inps) == 0:
        print('ERR: No jpg images found in ' + inp_path + '\n')
        quit()
    print('Found %d images!' % len(all_inps))

    all_features = []
    with torch.no_grad():
        textquery_feat = model.clip_model.encode_text(clip.tokenize("scene text").to(device))
        for i, img in enumerate(all_inps):
            image_input_source = preprocess(Image.open(img)).unsqueeze(0).to(device)
            features = model.evaluate(image_input_source)
            prompted_text_features = model.cross_attention_textquery(features.unsqueeze(1),
                                                                    textquery_feat.unsqueeze(1),
                                                                    textquery_feat.unsqueeze(1))[0].squeeze(1)

            all_features.append(features+prompted_text_features)
        image_features_source = torch.cat(all_features).to(device)
        prompts = model.prompt_learner(str_queries, str_queries,model.clip_model)
        text_features = model.text_encoder(prompts, model.prompt_learner.tokenized_prompts)

        image_features_source /= image_features_source.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_source = (model.logit_scale.exp() * text_features @ image_features_source.t()).softmax(dim=-1)

    print('Calculating mAP...\n')
    mAP = 0.
    for i in range(len(str_queries)):
        query = str_queries[i]
        rel = []
        for j in range(len(gt_data['data'][0, i][1])):
            rel.append(str(gt_data['data'][0, i][1][j][0][0]).lower())

        y_true = np.zeros((len(all_inps),))
        y_scores = np.zeros((len(all_inps),))
        for r in rel:
            y_true[all_inps.index(inp_path + r)] = 1
        y_scores = similarity_source[i].cpu().numpy()

        ap = average_precision_score(y_true, y_scores)
        mAP += ap

    mAP /= len(str_queries)
    print('Final mean Average Precision (mAP) = ' + str(mAP) + '\n')

model_path='./save/fdp-8th.pt'
gt_data_path = './datasets/IIIT_STR_V1.0/data.mat'
inp_path = './datasets/IIIT_STR_V1.0/imgDatabase_pad_square/'
test(model_path,gt_data_path,inp_path)