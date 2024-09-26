#%%
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import json
from bbox import get_union_bbox, draw_bboxes_with_labels
from hsg import Textual_HSG_new
from tqdm import tqdm
import inflect
p = inflect.engine()

import scipy.stats
import copy
from dataset import dataset_config, annot_loader, gpt_loader
from score import print_score

import torch
from torch.nn import functional as F
from pascal50s import Pascal50sDataset, calculate_pairwise_ranking
from glip_util import load_glip_with_sam
from mask_utils import get_image_parsing_graph, bboxes2masks, convert_oneformer_format_to_sam, merge_sam_and_oneformer, merge_glip_mask_to_img_all_masks, mask_nms, build_tree_with_root_glip, vis, get_segidx2bbox_from_masks
import ipdb
import cv2


from segment_anything import (
    sam_model_registry,
    #sam_hq_model_registry,
    SamPredictor
)

#%%
import argparse
parser = argparse.ArgumentParser(description='parse arguments')

parser.add_argument('--dataset', type=str, default='pascal50s', required=True, 
                    choices=['flickr8k_exp', 'composite', 'pascal50s', 'img_par_neg_obj', 'img_par_neg_rel', 'img_par_neg_attr', 'thumb', 'locnar1025', 'locnar_plausible'],
                    help='选择数据集 (默认: pascal50s)')
parser.add_argument('--glip_thr', type=int, default=0,
                    help='GLIP 阈值 (默认: 0)')
parser.add_argument('--embed_size', type=int, default=256, 
                    choices=[256, 768],
                    help='嵌入向量大小 (默认: 256)')
parser.add_argument('--model_type', type=str, default='pretrain', 
                    choices=['coco', 'pretrain', 'vg_dense'],
                    help='模型类型 (默认: pretrain)')
parser.add_argument('--total_parts', type=int, default=1,
                    help='总分片数 (默认: 1)')
parser.add_argument('--cur_part', type=int, default=0,
                    help='当前分片号 (默认: 3)')
parser.add_argument('--glob_img_feat', action='store_true', default=False,
                    help='是否使用额外的全图特征(默认: False)')
parser.add_argument('--split_cand_par', action='store_true', default=False,
                    help='是否将段落拆分成句子(默认: False)')

# 解析参数
args = parser.parse_args()

# dataset = 'pascal50s' # flickr8k_exp / composite / pascal50s / img_par_2182 / img_par_neg_rel /img_par_neg_attr
# glip_thr = 0
# embed_size = 256   # 256 / 768
# model_type = 'pretrain' # coco / pretrain / vg_dense
# total_parts = 4
# cur_part = 3

dataset = args.dataset
glip_thr = args.glip_thr
embed_size = args.embed_size
model_type = args.model_type
total_parts = args.total_parts
cur_part = args.cur_part
glob_img_feat = args.glob_img_feat
split_cand_par = args.split_cand_par
print(f'Done. dataset={dataset}, glip_thr={glip_thr}, embed_size={embed_size}, model_type={model_type}')

glip_res_path = f'blip_matching_results/{dataset}_modeltype-{model_type}_embsize-{embed_size}_globimgfeat-{glob_img_feat}_splitcand-{split_cand_par}_part{cur_part}.json'

#ipdb.set_trace()
from lavis.models import load_model_and_preprocess
device = "cuda" if torch.cuda.is_available() else "cpu"
model, vis_processors, txt_processors = load_model_and_preprocess(name="blip2", model_type=model_type, is_eval=True, device=device)
model_global, _, _ = load_model_and_preprocess(name="blip2", model_type='pretrain', is_eval=True, device=device)

# sam_checkpoint = '/home/lm1/projects/segment-anything/checkpoints/sam_vit_h_4b8939.pth'
# sam_predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=sam_checkpoint).to(device))

def extract_bbox_feats(raw_image, bboxes):
    if len(bboxes) == 0:
        return None
    sample = {"image": [], "text_input": []}
    for bbox in bboxes:
        bbox_image = raw_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        image = vis_processors["eval"](bbox_image).unsqueeze(0)
        sample['image'].append(image)

    sample['image'] = torch.vstack(sample['image']).to(device)
    features_image = model.extract_image_features_yzw(sample)
    return features_image

def extract_phrase_feats(phrases):
    if len(phrases) == 0:
        return None
    sample = {"image": [], "text_input": []}
    for i in range(len(phrases)):
        caption = phrases[i]
        text_input = txt_processors["eval"](caption)
        sample['text_input'].append(text_input)

    features_text = model.extract_text_features_yzw(sample)
    return features_text


def extract_phrase_feats_global(phrases):
    sample = {"image": [], "text_input": []}
    for i in range(len(phrases)):
        caption = phrases[i]
        text_input = txt_processors["eval"](caption)
        sample['text_input'].append(text_input)

    features_text = model_global.extract_text_features_yzw(sample)
    return features_text
#%%
images, candidates, refs, human_scores, human_scores_list, categories = annot_loader(dataset)
gpt_res = gpt_loader(dataset)
gpt_res_new = {}
for k, v in gpt_res.items():
    gpt_res_new[' '.join(k.split())] = v 
gpt_res = gpt_res_new

glip_mask_dir = dataset_config[dataset]['glip_mask_dir']
sam_mask_dir = dataset_config[dataset]['sam_mask_dir']
oneformer_mask_dir = dataset_config[dataset]['oneformer_mask_dir']
saliency_map_dir = dataset_config[dataset]['saliency_map_dir']
glob_img_feat_dir = dataset_config[dataset]['glob_img_feat_dir']


#%%
part_images_num = int(len(images) / total_parts)
bgn_idx = part_images_num * cur_part
end_idx = len(images) if cur_part==total_parts-1 else part_images_num * (cur_part+1)
print(f"part {cur_part/total_parts}: bgn-{bgn_idx}, end-{end_idx}")
#%%
res_json = {}
#for idx in range(6):
#for idx in [1372]:
#for idx in tqdm(range(len(images))):
for idx in tqdm(range(bgn_idx, end_idx)):
    ################################################################################################
    # input
    image_path = images[idx]
    image_name = image_path.split('/')[-1]
    image_prefix = image_path.split('/')[-1].split('.')[0]
    one_cand = candidates[idx]
    #one_human_score = human_scores[idx]
    #print('idx', idx, image_path, one_human_score)
    #print(one_cand)
    one_cand = ' '.join(one_cand.split())
    gpt_result = copy.deepcopy(gpt_res[one_cand])
    _, candidate_objs, phrase_ids = Textual_HSG_new(one_cand, gpt_result)
    #print(gpt_result)
    saliency_map = np.load(os.path.join(saliency_map_dir, f'{image_prefix}.npy'), allow_pickle=True)

    glip_masks = np.load(os.path.join(glip_mask_dir, f'{image_prefix}.npy'), allow_pickle=True).item()
    glip_masks_new = {}
    for k, v in glip_masks.items():
        glip_masks_new[' '.join(k.split())] = v 
    glip_masks = glip_masks_new[one_cand]

    if os.path.exists(f'/home/lm1/projects/HSG/cap_eval_with_grounding/filtered_masks_with_glip/{dataset}/{idx}.npy'):
        filtered_masks_with_glip = np.load(f'/home/lm1/projects/HSG/cap_eval_with_grounding/filtered_masks_with_glip/{dataset}/{idx}.npy', allow_pickle=True)
    else: 
        print(f"generate filtered_masks_with_glip to {dataset}/{idx}.npy")
        sam_masks = np.load(os.path.join(sam_mask_dir, f'{image_name}_sam_result.npy'), allow_pickle=True).item()[image_name]
        oneformer_masks = np.load(os.path.join(oneformer_mask_dir, f'{image_prefix}.npy'), allow_pickle=True).item()[image_prefix]
        oneformer_masks = convert_oneformer_format_to_sam(oneformer_masks) 
        all_masks = merge_sam_and_oneformer(sam_masks, oneformer_masks, same_threshold=0.85)
        filtered_masks, keep_indices = mask_nms(all_masks, threshold=0.9)
        filtered_masks_with_glip = merge_glip_mask_to_img_all_masks(glip_masks, filtered_masks, same_threshold=0.85)
        np.save(f'/home/lm1/projects/HSG/cap_eval_with_grounding/filtered_masks_with_glip/{dataset}/{idx}.npy', filtered_masks_with_glip)

    # print('all masks', len(all_masks))
    # print('filtered_masks', len(filtered_masks))
    # print('all_masks_with_glip', len(filtered_masks_with_glip))
    # print(glip_masks['grounded_phrase_entities'])

    grounded_phrases, grounded_phrase_idxs, grounded_union_bboxes, grounded_all_bboxes = load_glip_with_sam(glip_masks)

    #print("segidx2bbox")
    segidx2bbox = get_segidx2bbox_from_masks(filtered_masks_with_glip)
    grounded_seg_idxs = []
    for grounded_i in range(len(grounded_phrases)):
        one_union_bbox = grounded_union_bboxes[grounded_i]
        one_all_bboxes = grounded_all_bboxes[grounded_i]
        one_grounded_seg_idxs = []
        for one in one_all_bboxes:
            for x in segidx2bbox:
                if one in segidx2bbox[x]:
                    one_grounded_seg_idxs.append(x)
        #print(grounded_phrases[grounded_i], one_grounded_seg_idxs)
        grounded_seg_idxs.append(one_grounded_seg_idxs)

    root_node = build_tree_with_root_glip(filtered_masks_with_glip, image_path, global_discard_threshold=0.03, discard_threshold=0.05, contain_threshold=0.75)
    #vis(root_node, filtered_masks_with_glip, image_path)

    id2info = {}
    root_node.walking_through_tree(None, id2info)
    raw_image = Image.open(image_path).convert("RGB")
    width = raw_image.size[0]
    height = raw_image.size[1]
    id2info[-1]['area'] = width * height

    # empty_pos = [i for i in range(len(grounded_bboxes)) if grounded_bboxes[i]==[]]
    # new_grounded_phrases = [one_cand] + [grounded_phrases[x] for x in range(len(grounded_phrases)) if x not in empty_pos]
    # new_grounded_bboxes = [[0, 0, raw_image.size[0], raw_image.size[1]]] + [grounded_bboxes[x][0] for x in range(len(grounded_phrases)) if x not in empty_pos]

    empty_pos = [x for x in range(len(candidate_objs)) if  x not in grounded_phrase_idxs]

    glob_cands = [x.strip() for x in one_cand.split('.') if x.strip()!=''] if split_cand_par else [one_cand] 

    if glob_img_feat:   # still need to extract global text feats 
        global_image_feature = np.load(os.path.join(glob_img_feat_dir, f'{image_name}.npy'), allow_pickle=True).item()
        new_grounded_phrases = grounded_phrases
        new_grounded_bboxes = grounded_union_bboxes
    else:
        new_grounded_phrases = [one_cand] + grounded_phrases
        new_grounded_bboxes = [[0, 0, raw_image.size[0], raw_image.size[1]]] + grounded_union_bboxes


    saliency_map_resized = cv2.resize(saliency_map,(width,height))
    saliency_weights = []
    for one_mask in filtered_masks_with_glip:
        # 使用mask来选择saliency map中相应的区域
        mask = one_mask['segmentation']
        if np.all(mask == False):
            saliency_weights.append(0.0)
            continue

        masked_saliency = saliency_map_resized * mask

        # 计算该区域的平均saliency值
        average_saliency = np.mean(masked_saliency[mask > 0])
        saliency_weights.append(average_saliency.item())
    #saliency_weights = torch.tensor(saliency_weights).to(device)

    #ipdb.set_trace()
    features_image = extract_bbox_feats(raw_image, new_grounded_bboxes)
    features_text = extract_phrase_feats(new_grounded_phrases)

    if features_image==None:
        itm_similarities = []
        itc_similarities = []
    else:
        if embed_size == 256:
            candidate_obj_feats_normed = features_text['text_embeds_proj']
            image_feats_normed = features_image['image_embeds_proj']

        elif embed_size == 768:
            candidate_obj_feats_normed = features_text['text_embeds']
            image_feats_normed = features_image['image_embeds']
            candidate_obj_feats_normed = F.normalize(candidate_obj_feats_normed, dim=-1)
            image_feats_normed = F.normalize(image_feats_normed, dim=-1)
        else:
            raise Exception(f"Unexpected embed_size={embed_size}")

        candidate_obj_feats_normed = candidate_obj_feats_normed[:,0,:].unsqueeze(2)

        itc_similarities = torch.bmm(image_feats_normed, candidate_obj_feats_normed).squeeze(2).max(dim=1)[0]

        image_inputs = features_image['vit_feat'].to(model.device)
        text_ids = features_text['text_ids']
        text_atts = features_text['text_atts']
        with torch.no_grad():
            itm_output = model.compute_itm_yzw(
                image_inputs=image_inputs,
                text_ids=text_ids,
                text_atts=text_atts
            ).float()
        itm_similarities = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]

        itm_similarities = itm_similarities.tolist()
        itc_similarities = itc_similarities.tolist()

    if glob_img_feat:
        global_features_text = extract_phrase_feats_global(glob_cands)
        if embed_size == 256:
            global_candidate_obj_feats_normed = global_features_text['text_embeds_proj']
            global_image_feats_normed = global_image_feature['image_embeds_proj'].repeat(len(glob_cands), 1, 1).to(model.device)

        elif embed_size == 768:
            global_candidate_obj_feats_normed = global_features_text['text_embeds']
            global_image_feats_normed = global_image_feature['image_embeds'].repeat(len(glob_cands), 1, 1).to(model.device)

            global_candidate_obj_feats_normed = F.normalize(global_candidate_obj_feats_normed, dim=-1)
            global_image_feats_normed = F.normalize(global_image_feats_normed, dim=-1)
        else:
            raise Exception(f"Unexpected embed_size={embed_size}")

        global_candidate_obj_feats_normed = global_candidate_obj_feats_normed[:,0,:].unsqueeze(2)
        global_itc_similarities = torch.bmm(global_image_feats_normed, global_candidate_obj_feats_normed).squeeze(2).max(dim=1)[0]

        global_image_inputs = global_image_feature['vit_feat'].repeat(len(glob_cands), 1, 1).to(model.device)
        global_text_ids = global_features_text['text_ids']
        global_text_atts = global_features_text['text_atts']
        with torch.no_grad():
            itm_output = model_global.compute_itm_yzw(
                image_inputs=global_image_inputs,
                text_ids=global_text_ids,
                text_atts=global_text_atts
            ).float()
        global_itm_similarities = torch.nn.functional.softmax(itm_output, dim=1)[:, 1]

        global_itc_similarity = global_itc_similarities.mean().tolist()
        global_itm_similarity = global_itm_similarities.mean().tolist()

        itm_similarities = [global_itm_similarity] + itm_similarities
        itc_similarities = [global_itc_similarity] + itc_similarities

    #print("human score", human_scores[idx],  grounded_seg_idxs)
    #ipdb.set_trace()

    res_json[idx] = {'itc':itc_similarities, 'itm':itm_similarities, 
                     'id2info':id2info, 'grounded_seg_idxs':grounded_seg_idxs, 'saliency_weights':saliency_weights,
                     'human_score': human_scores_list[idx] if human_scores_list else human_scores[idx], 
                     'phrases': candidate_objs, 'phrase_ids':phrase_ids, 
                     'grounded_phrases':grounded_phrases, 'grounded_phrase_idxs':grounded_phrase_idxs,
                     'image_path':image_path, 'candidate':one_cand, 'category': categories[idx] if categories else ''}
    # print(f'itm global: {itm_similarities[0]:.8f}')
    # print(f'itc global: {itc_similarities[0]:.8f}')

#ipdb.set_trace()
#%%
with open(glip_res_path, 'w') as f:
    json.dump(res_json, f, indent=4)

with open(glip_res_path, 'r') as f:
    res_data = json.load(f)
res_data = {int(k):v for k,v in res_data.items()}
for idx in res_data:
    res_data[idx]['id2info'] = {int(k):v for k,v in res_data[idx]['id2info'].items()}

print_score(res_data, dataset)

print(f'Done. dataset={dataset}, glip_thr={glip_thr}, embed_size={embed_size}, model_type={model_type}')

# %%
# visualize
# img = Image.open(image_path)

# bboxes = glip_gpt_result[one_cand][1]['bboxes'].tolist()
# labels = glip_gpt_result[one_cand][1]['labels'].tolist()
# labels = [str(x) for x in labels]
# img_output = draw_bboxes_with_labels(image_path, bboxes, labels)
# display(img_output)