import torch
import clip
import torch.nn as nn
from Transformer_Modules import MultiHeadAttention
import torch.nn.functional as F
from utils import Kmeans_cluster
device = "cuda" if torch.cuda.is_available() else "cpu"

class TextCLIP(nn.Module):
    def __init__(self, device, model):
        super(TextCLIP, self).__init__()
        self.model = model
        self.device = device

    def forward(self, text):
        return self.model.encode_text(text).to(self.device)

class ImageCLIP(nn.Module):
    def __init__(self, device, model):
        super(ImageCLIP, self).__init__()
        self.model = model
        self.device = device

    def forward(self, image):
        return self.model.encode_image(image).to(self.device)

class PromptTextEncoder(nn.Module):
    def __init__(self, device, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.device = device

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        x = x.to(self.device)
        return x

class PromptLearner(nn.Module):
    def __init__(self, device, clip_model):
        super().__init__()
        n_ctx1 = 2
        n_ctx2 = 4
        self.dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = 224
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        self.device = device

        ctx_vectors1 = torch.empty(n_ctx1, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors1, std=0.02)
        self.ctx1 = nn.Parameter(ctx_vectors1)

        ctx_vectors2 = torch.empty(n_ctx2, ctx_dim, dtype=self.dtype)
        nn.init.normal_(ctx_vectors2, std=0.02)
        self.ctx2 = nn.Parameter(ctx_vectors2)

        self.n_ctx1 = n_ctx1
        self.n_ctx2 = n_ctx2

        self.kmeans, self.function_cluster, self.content_cluster = Kmeans_cluster()

    def forward(self, classnames, queries, clip_model):
        group = int(len(queries) / len(classnames))
        prompts = []
        tokenized_prompts = []
        prompt_prefix1 = " ".join(["X"] * self.n_ctx1)
        prompt_prefix2 = " ".join(["X"] * self.n_ctx2)

        ctx1 = self.ctx1.unsqueeze(0).expand(group, -1, -1).to(self.device)
        ctx2 = self.ctx2.unsqueeze(0).expand(group, -1, -1).to(self.device)

        for ni in range(len(classnames)):
            classname = classnames[ni]
            with torch.no_grad():
                classname_tensor=clip_model.encode_text(clip.tokenize(classname).to(device))
                predicted_cluster = self.kmeans.predict(classname_tensor.cpu())
            if predicted_cluster == self.function_cluster:
                pts = [prompt_prefix1 + " " + name + "." for name in queries[ni * group:(ni + 1) * group]]
                tokenized_prompt = torch.cat([clip.tokenize(p) for p in pts]).to(self.device)
                tokenized_prompts.append(tokenized_prompt)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokenized_prompt).type(self.dtype)
                prefix = embedding[:, :1, :]
                suffix = embedding[:, 1 + self.n_ctx1:, :]
                prompts.append(torch.cat([prefix, ctx1, suffix], dim=1))
            else:
                pts = [prompt_prefix2 + " " + name + "." for name in queries[ni * group:(ni + 1) * group]]
                tokenized_prompt = torch.cat([clip.tokenize(p) for p in pts]).to(self.device)
                tokenized_prompts.append(tokenized_prompt)
                with torch.no_grad():
                    embedding = clip_model.token_embedding(tokenized_prompt).type(self.dtype)
                prefix = embedding[:, :1, :]
                suffix = embedding[:, 1 + self.n_ctx2:, :]
                prompts.append(torch.cat([prefix, ctx2, suffix], dim=1))
        prompts = torch.cat(prompts, dim=0)
        self.tokenized_prompts = torch.cat(tokenized_prompts, dim=0)
        return prompts

class CustomCLIP(nn.Module):
    def __init__(self, device, clip_model):
        super().__init__()
        self.clip_model = clip_model
        self.prompt_learner = PromptLearner(device, clip_model)
        self.image_encoder = clip_model.visual
        self.text_encoder = PromptTextEncoder(device, clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

        self.n_head = 8
        self.d_k = 1024
        self.d_v = 1024
        self.cross_attention_textquery = MultiHeadAttention(self.n_head, self.d_k, self.d_v).cuda()

        self.spacial_dim = 640 // 32
        self.new_positional_embedding = nn.Parameter(torch.randn(self.spacial_dim ** 2 + 1, 2048) / 2048 ** 0.5)  # (50,2048)=》(400,2048)
        params = nn.functional.interpolate(self.clip_model.visual.attnpool.positional_embedding.unsqueeze(0).unsqueeze(0)
            , size=(self.spacial_dim ** 2 + 1, 2048), mode='nearest').squeeze()
        self.new_positional_embedding = nn.Parameter(params)

        self.new_v_proj = nn.Conv2d(2048, 2048, 1).cuda()
        self.new_c_proj = nn.Conv2d(2048, 1024, 1).cuda()
        loaded = torch.load('./clip_reformulated/RN50_reformulated.pth', map_location='cuda')
        self.new_v_proj.load_state_dict(loaded['v_proj'], strict=False)
        self.new_c_proj.load_state_dict(loaded['c_proj'], strict=False)

        self.in_channels = 1024
        self.binarize = nn.Conv2d(1024, 1, 1).cuda()

    def stem(self, x):
        x = self.clip_model.visual.relu1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
        x = self.clip_model.visual.relu2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
        x = self.clip_model.visual.relu3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
        x = self.clip_model.visual.avgpool(x)
        return x

    def candidates_similarity(self, logit_scale, image_features, candidates_text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        candidates_text_features = candidates_text_features / candidates_text_features.norm(dim=-1, keepdim=True)
        candidates_text_features = candidates_text_features.view(-1, 5, 1024)
        candidates_similarity = []
        logit_scale = logit_scale.exp()
        for ii in range(image_features.shape[0]):
            logits = (logit_scale * image_features[ii, :].unsqueeze(0) @ candidates_text_features[ii, :, :].t())
            candidates_similarity.append(logits.squeeze(0))
        candidates_similarity = torch.stack(candidates_similarity).to(image_features.device)
        return candidates_similarity

    def cosine_similarity(self, logit_scale, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

    def forward(self, classnames, image, all_candidates):
        layer0_feat = self.stem(image.type(self.dtype)).to(device)
        x = self.clip_model.visual.layer1(layer0_feat)
        x = self.clip_model.visual.layer2(x)
        x = self.clip_model.visual.layer3(x)
        layer4_feat = self.clip_model.visual.layer4(x)
        x = layer4_feat.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.new_positional_embedding[:, None, :].to(x.dtype)

        prompts = self.prompt_learner(classnames, classnames,self.clip_model)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        global_text_features = self.text_encoder(prompts, tokenized_prompts).to(device)
        candidates_prompts = self.prompt_learner(classnames, all_candidates,self.clip_model)
        tokenized_candidates_prompts = self.prompt_learner.tokenized_prompts
        candidates_text_features = self.text_encoder(candidates_prompts, tokenized_candidates_prompts).to(device)
        textquery_features = self.clip_model.encode_text(clip.tokenize("scene text").to(device)).repeat(layer4_feat.size(0), 1)

        new_v = self.new_v_proj(layer4_feat)
        new_global_features = self.new_c_proj(new_v)
        score_map = self.binarize(new_global_features).squeeze(1)
        new_attn_mask = torch.sigmoid(score_map).flatten(start_dim=1)
        new_attn_mask = torch.cat([torch.ones(score_map.size(0), 1).cuda(), new_attn_mask],dim=1)
        new_attn_mask = new_attn_mask.unsqueeze(-1).permute(0, 2, 1).repeat(self.clip_model.visual.attnpool.num_heads,1, 1)

        x_new, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.clip_model.visual.attnpool.num_heads,
            q_proj_weight=self.image_encoder.attnpool.q_proj.weight,
            k_proj_weight=self.image_encoder.attnpool.k_proj.weight,
            v_proj_weight=self.image_encoder.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.image_encoder.attnpool.q_proj.bias, self.image_encoder.attnpool.k_proj.bias,
                                    self.image_encoder.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.image_encoder.attnpool.c_proj.weight,
            out_proj_bias=self.image_encoder.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=False,
            need_weights=False,
            attn_mask=new_attn_mask
        )

        global_features = x_new.squeeze(0)
        prompted_text_features = self.cross_attention_textquery(global_features.unsqueeze(1),
                                                                textquery_features.unsqueeze(1),
                                                                textquery_features.unsqueeze(1))
        prompted_text_features = prompted_text_features[0].squeeze(1)
        fused_features = global_features + prompted_text_features
        candidates_features_logits = self.candidates_similarity(self.logit_scale, fused_features,candidates_text_features)
        fused_features_logits, query_features_logits = self.cosine_similarity(self.logit_scale, fused_features,global_text_features)
        return fused_features_logits, query_features_logits, candidates_features_logits, score_map

    def evaluate(self, image):
        layer0_feat = self.stem(image.type(self.dtype)).to(device)
        x = self.clip_model.visual.layer1(layer0_feat)
        x = self.clip_model.visual.layer2(x)
        x = self.clip_model.visual.layer3(x)
        layer4_feat = self.clip_model.visual.layer4(x)
        x = layer4_feat.flatten(start_dim=2).permute(2, 0, 1)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)
        x = x + self.new_positional_embedding[:, None, :].to(x.dtype)

        new_v = self.new_v_proj(layer4_feat)
        new_global_features = self.new_c_proj(new_v)
        score_map = self.binarize(new_global_features).squeeze(1)
        new_attn_mask = torch.sigmoid(score_map).flatten(start_dim=1)
        new_attn_mask = torch.cat([torch.ones(score_map.size(0), 1).cuda(), new_attn_mask],dim=1)
        new_attn_mask = new_attn_mask.unsqueeze(-1).permute(0, 2, 1).repeat(self.clip_model.visual.attnpool.num_heads,1, 1)

        x_new, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.clip_model.visual.attnpool.num_heads,
            q_proj_weight=self.image_encoder.attnpool.q_proj.weight,
            k_proj_weight=self.image_encoder.attnpool.k_proj.weight,
            v_proj_weight=self.image_encoder.attnpool.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.image_encoder.attnpool.q_proj.bias, self.image_encoder.attnpool.k_proj.bias,
                                    self.image_encoder.attnpool.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.image_encoder.attnpool.c_proj.weight,
            out_proj_bias=self.image_encoder.attnpool.c_proj.bias,
            use_separate_proj_weight=True,
            training=False,
            need_weights=False,
            attn_mask=new_attn_mask
        )

        global_features = x_new.squeeze(0)
        return global_features