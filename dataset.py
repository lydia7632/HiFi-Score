from pascal50s import Pascal50sDataset
from tqdm import tqdm
import json
import numpy as np
import os

dataset_config = {
  "pascal50s": {
    "gpt_txt": "/home/lm1/projects/HSG/hsg/pascal50s_hsg_by_gpt_v2.txt",
    "glip_mask_dir": "/home/lm1/projects/HSG/cap_eval_with_grounding/glip_res_with_sam/pascal50s",
    "sam_mask_dir": "/home/lm1/projects/HSG/sam_results/masks/Pascal50S",
    "oneformer_mask_dir": "/home/lm1/projects/HSG/oneformer_results/pascal50s",
    "saliency_map_dir": "/home/lm1/projects/HSG/saliency_results/pascal50s",
    "glob_img_feat_dir":"/home/lm1/projects/HSG/cap_eval_with_grounding/img_feats/pascal50s" 
  },
  "flickr8k_exp": {
    "input_json": "/home/lm1/datasets/Flickr8K/flickr8k.json",
    "image_directory": "/home/lm1/datasets/Flickr8K/",
    "gpt_txt": "/home/lm1/projects/HSG/hsg/flickr8k_hsg_by_gpt_v2.txt",
    "glip_mask_dir": "/home/lm1/projects/HSG/cap_eval_with_grounding/glip_res_with_sam/flickr8k_exp",
    "sam_mask_dir": "/home/lm1/projects/HSG/sam_results/masks/Flickr8K",
    "oneformer_mask_dir": "/home/lm1/projects/HSG/oneformer_results/flickr8k_exp",
    "saliency_map_dir": "/home/lm1/projects/HSG/saliency_results/flickr8k",
    "glob_img_feat_dir":"/home/lm1/projects/HSG/cap_eval_with_grounding/img_feats/flickr8k" 
  },
  "composite": {
    "input_json": "/home/lm1/datasets/Composite_Dataset/composite.json",
    "image_directory": "/home/lm1/datasets/Composite_Dataset/all_images",
    "gpt_txt": "/home/lm1/projects/HSG/hsg/composite_hsg_by_gpt_v2.txt",
    "glip_mask_dir": "/home/lm1/projects/HSG/cap_eval_with_grounding/glip_res_with_sam/composite",
    "sam_mask_dir": "/home/lm1/projects/HSG/sam_results/masks/Composite",
    "oneformer_mask_dir": "/home/lm1/projects/HSG/oneformer_results/composite",
    "saliency_map_dir": "/home/lm1/projects/HSG/saliency_results/composite",
    "glob_img_feat_dir":"/home/lm1/projects/HSG/cap_eval_with_grounding/img_feats/composite" 
  },
  "img_par_neg_obj": {
    "input_json": "/home/lm1/projects/HSG/long_context_dataset/img_par_with_neg_obj_2182.json",
    "image_directory": "/home/lm1/datasets/vg/images",
    "gpt_txt": "/home/lm1/projects/HSG/long_context_dataset/img_par_with_neg_obj_by_gpt.txt",
    "glip_mask_dir": "/home/lm1/projects/HSG/cap_eval_with_grounding/glip_res_with_sam/img_par_neg_obj",
    "sam_mask_dir": "/home/lm1/projects/HSG/sam_results/masks/ImageParagraphs",
    "oneformer_mask_dir": "/home/lm1/projects/HSG/oneformer_results/ImageParagraphs",
    "saliency_map_dir": "/home/lm1/projects/HSG/saliency_results/ImageParagraphs",
    "glob_img_feat_dir":"/home/lm1/projects/HSG/cap_eval_with_grounding/img_feats/ImageParagraphs" 
  },
  "img_par_neg_rel": {
    "input_json": "/home/lm1/projects/HSG/long_context_dataset/img_par_with_neg_rel.json",
    "image_directory": "/home/lm1/datasets/vg/images",
    "gpt_txt": "/home/lm1/projects/HSG/long_context_dataset/img_par_neg_rel_hsg_by_gpt_v3_json.txt",
    "glip_mask_dir": "/home/lm1/projects/HSG/cap_eval_with_grounding/glip_res_with_sam/img_par_neg_rel",
    "sam_mask_dir": "/home/lm1/projects/HSG/sam_results/masks/ImageParagraphs",
    "oneformer_mask_dir": "/home/lm1/projects/HSG/oneformer_results/ImageParagraphs",
    "saliency_map_dir": "/home/lm1/projects/HSG/saliency_results/ImageParagraphs",
    "glob_img_feat_dir":"/home/lm1/projects/HSG/cap_eval_with_grounding/img_feats/ImageParagraphs" 
  },
  "img_par_neg_attr": {
    "input_json": "/home/lm1/projects/HSG/long_context_dataset/img_par_with_neg_attr.json",
    "image_directory": "/home/lm1/datasets/vg/images",
    "gpt_txt": "/home/lm1/projects/HSG/long_context_dataset/img_par_neg_attr_hsg_by_gpt_v3_json.txt",
    "glip_mask_dir": "/home/lm1/projects/HSG/cap_eval_with_grounding/glip_res_with_sam/img_par_neg_attr",
    "sam_mask_dir": "/home/lm1/projects/HSG/sam_results/masks/ImageParagraphs",
    "oneformer_mask_dir": "/home/lm1/projects/HSG/oneformer_results/ImageParagraphs",
    "saliency_map_dir": "/home/lm1/projects/HSG/saliency_results/ImageParagraphs",
    "glob_img_feat_dir":"/home/lm1/projects/HSG/cap_eval_with_grounding/img_feats/ImageParagraphs" 
  },
  "thumb": {
    "input_json": "/home/lm1/projects/HSG/long_context_dataset/thumb.json",
    "image_directory": "/home/lm1/datasets/THumB/mscoco/THumB_images",
    "gpt_txt": "/home/lm1/projects/HSG/long_context_dataset/thumb_hsg_by_gpt_v3_json.txt",
    "glip_mask_dir": "/home/lm1/projects/HSG/cap_eval_with_grounding/glip_res_with_sam/thumb",
    "sam_mask_dir": "/home/lm1/projects/HSG/sam_results/masks/thumb",
    "oneformer_mask_dir": "/home/lm1/projects/HSG/oneformer_results/thumb",
    "saliency_map_dir": "/home/lm1/projects/HSG/saliency_results/thumb",
    "glob_img_feat_dir":"/home/lm1/projects/HSG/cap_eval_with_grounding/img_feats/thumb" 
  }
}


def annot_loader(dataset):
    if dataset == 'pascal50s':
        images, candidates, refs, human_scores, categories= pascal50s_annot_loader()
        return images, candidates, refs, human_scores, None, categories
    elif dataset == 'flickr8k_exp':
        images, candidates, refs, human_scores, human_scores_list = flickr8k_exp_annot_loader(dataset_config['flickr8k_exp']['input_json'], dataset_config['flickr8k_exp']['image_directory'])
        return images, candidates, refs, human_scores, human_scores_list, None
    elif dataset == 'thumb':
        images, candidates, refs, human_scores = thumb_annot_loader(dataset_config[dataset]['input_json'], dataset_config[dataset]['image_directory'])
        return images, candidates, refs, human_scores, None, None
    elif 'locnar' in dataset:
        images, candidates, refs, human_scores = locnar_dataset_annot_loader(dataset_config[dataset]['input_json'], dataset_config[dataset]['image_directory'])
        return images, candidates, refs, human_scores, None, None
    else:
        images, candidates, refs, human_scores = caption_dataset_annot_loader(dataset_config[dataset]['input_json'], dataset_config[dataset]['image_directory'])
        return images, candidates, refs, human_scores, None, None


def pascal50s_annot_loader():
    Pascal50s_Dataset = Pascal50sDataset()
    images = []
    candidates = []
    categories = []
    human_labels = []
    human_scores = [] # double human_label, consistent with other datasets
    refs = []
    for i in tqdm(range(len(Pascal50s_Dataset))):
        img_path, a, b, references, category, label = Pascal50s_Dataset[i]
        images.append(img_path)
        images.append(img_path)
        candidates.append(a)
        candidates.append(b)
        refs.append([' '.join(x.split()) for x in references])
        refs.append([' '.join(x.split()) for x in references])
        categories.append(int(category))
        categories.append(int(category))
        #human_labels.append(label)
        human_scores.append(label)
        human_scores.append(label)
    print("human_labels", len(human_labels))
    print("candidates", len(candidates))
    return images, candidates, refs, human_scores, categories


def flickr8k_exp_annot_loader(input_json, image_directory):
    data = {}
    with open(input_json, 'r') as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    images = []
    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        if v['image_path'] == 'dfb4dded9ea53d8d.jpg':
            continue
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue
            images.append(os.path.join(image_directory, v['image_path']))
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))

            human_scores.append(human_judgement['rating'])

    print("human_scores", len(human_scores))
    print("candidates", len(candidates))

    new_images = []
    new_refs = []
    new_candidates = []
    new_human_scores = []
    human_scores_list = []
    for m in range(int(len(human_scores)/3)):
        if candidates[m*3]==candidates[m*3+1]==candidates[m*3+2] and refs[m*3]==refs[m*3+1]==refs[m*3+2]:
            pass
        else:
            print(m, 'Error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        new_images.append(images[m*3])
        new_refs.append(refs[m*3])
        new_candidates.append(candidates[m*3])
        new_human_scores.append((human_scores[m*3]+human_scores[m*3+1]+human_scores[m*3+2])/3)
        human_scores_list.append([human_scores[m*3], human_scores[m*3+1], human_scores[m*3+2]])

    images = new_images
    refs = new_refs
    candidates = new_candidates
    old_human_scores = human_scores
    human_scores = new_human_scores

    print("After merge:")
    print("human_scores", len(human_scores))
    print("candidates", len(candidates))
    print("refs", len(refs))
    return images, candidates, refs, human_scores, human_scores_list


def thumb_annot_loader(input_json, image_directory):
    data = {}
    with open(input_json, 'r') as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    images = []
    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        for human_judgement in v['human_judgement']:
            images.append(os.path.join(image_directory, v['image_path']))
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))
            human_scores.append(human_judgement['rating'])

    print("human_scores", len(human_scores))
    print("candidates", len(candidates))
    return images, candidates, refs, human_scores


def caption_dataset_annot_loader(input_json, image_directory):
    data = {}
    with open(input_json, 'r') as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    images = []
    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        if v['image_path'] == 'dfb4dded9ea53d8d.jpg':
            continue
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue
            images.append(os.path.join(image_directory, v['image_path']))
            refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))

            human_scores.append(human_judgement['rating'])

    print("human_scores", len(human_scores))
    print("candidates", len(candidates))
    return images, candidates, refs, human_scores


def locnar_dataset_annot_loader(input_json, image_directory):
    data = {}
    with open(input_json, 'r') as f:
        data.update(json.load(f))
    print('Loaded {} images'.format(len(data)))

    images = []
    refs = []
    candidates = []
    human_scores = []
    for k, v in list(data.items()):
        if v['image_path'] in ['dfb4dded9ea53d8d.jpg', 'eb54c9abbe04799e.jpg', 'd7f94c3dbd68a87e.jpg', '1698ab61bc5c955f.jpg', '67103d083d358b64.jpg']:
            continue
        for human_judgement in v['human_judgement']:
            if np.isnan(human_judgement['rating']):
                print('NaN')
                continue
            images.append(os.path.join(image_directory, v['image_path']))
            #refs.append([' '.join(gt.split()) for gt in v['ground_truth']])
            candidates.append(' '.join(human_judgement['caption'].split()))

            human_scores.append(human_judgement['rating'])

    print("human_scores", len(human_scores))
    print("candidates", len(candidates))
    return images, candidates, refs, human_scores


def gpt_loader(dataset):
    gpt_res = {}
    with open(dataset_config[dataset]['gpt_txt'], 'r', encoding='utf-8') as f:
        results = f.readlines()
    for one in results:
        gpt_res.update(json.loads(one))
    print('GPT file {} loaded'.format(dataset_config[dataset]['gpt_txt']))
    return gpt_res