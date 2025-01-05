import numpy as np
from bbox import get_union_bbox
import os
import inflect
p = inflect.engine()
from mask_utils import bboxes2masks

import torch
from tqdm import tqdm

from segment_anything import (
    sam_model_registry,
    SamPredictor
)

def load_glip_result(result_path, one_cand, glip_thr=0):
    glip_gpt_result = np.load(result_path, allow_pickle=True).item()
    if len(glip_gpt_result[one_cand]) == 0:
        print("len(glip_res)=0", result_path)
        return [], []

    grounded_phrases = []
    grounded_bboxes = []
    for phrase_i in range(len(glip_gpt_result[one_cand])):
        phrase_result = glip_gpt_result[one_cand][phrase_i]
        phrase = phrase_result['caption']
        entities = phrase_result['entities']
        bboxes = phrase_result['bboxes'].tolist()
        labels = phrase_result['labels'].tolist()
        scores = phrase_result['scores'].tolist()
        #print("\nphrase:", phrase)

        phrase_bbox = []
        phrase_bboxes = []
        for ett_i, one_ett in enumerate(entities):
            ett_idx = ett_i + 1
            ett_pos = [i for i in range(len(labels)) if labels[i]==ett_idx and scores[i]>glip_thr]
            ett_bboxes = [bboxes[x] for x in ett_pos]
            ett_scores = [scores[x] for x in ett_pos]
            #print('\t', one_ett, len(ett_bboxes), ett_scores, end='')
            if ett_bboxes:
                if p.singular_noun(one_ett): # plural
                    #print(' plural')
                    ett_bbox = get_union_bbox(ett_bboxes)
                else:
                    #print(' singular')
                    max_score_pos = ett_scores.index(max(ett_scores))
                    ett_bbox = ett_bboxes[max_score_pos]
                phrase_bboxes.append(ett_bbox)
        #ipdb.set_trace()

        if phrase_bboxes:
            phrase_bbox.append(get_union_bbox(phrase_bboxes))

        if phrase_bbox:
            grounded_bboxes.append(phrase_bbox)
            grounded_phrases.append(phrase)

    return grounded_bboxes, grounded_phrases

def load_glip_with_sam(glip_masks):

    grounded_phrases = []
    grounded_phrase_idxs = []
    grounded_union_bboxes = []
    grounded_all_bboxes = []
    for phrase_i in range(len(glip_masks['grounded_phrases'])):
        one_bboxes = glip_masks['grounded_bboxes'][phrase_i]
        one_scores = glip_masks['grounded_scores'][phrase_i]
        one_masks = glip_masks['grounded_phrase_masks'][phrase_i]
        one_phrase = glip_masks['grounded_phrases'][phrase_i]
        one_phrase_entities = glip_masks['grounded_phrase_entities'][phrase_i]

        ett_union_bboxes = []
        ett_all_bboxes = []
        #print(one_phrase_entities)
        for ett_i in range(len(one_phrase_entities)):
            one_ett = one_phrase_entities[ett_i]
            ett_bboxes = one_bboxes[ett_i]
            ett_scores = one_scores[ett_i]
            if ett_bboxes:
                if p.singular_noun(one_ett): # plural
                    #print(' plural', one_ett, len(ett_bboxes))
                    ett_union_bboxes.append(get_union_bbox(ett_bboxes))
                    ett_all_bboxes += ett_bboxes
                else:
                    #print(' singular', one_ett, len(ett_bboxes))
                    max_score_pos = ett_scores.index(max(ett_scores))
                    ett_union_bboxes.append(ett_bboxes[max_score_pos])
                    ett_all_bboxes += [ett_bboxes[max_score_pos]]

        if ett_union_bboxes:
            grounded_phrases.append(one_phrase)
            grounded_phrase_idxs.append(phrase_i)
            grounded_union_bboxes.append(get_union_bbox(ett_union_bboxes))
            grounded_all_bboxes.append(ett_all_bboxes)

    return grounded_phrases, grounded_phrase_idxs, grounded_union_bboxes, grounded_all_bboxes


def glip_with_sam(images, image_dir, output_dir):
    device = 'cuda'
    sam_checkpoint = '/home/lm1/projects/segment-anything/checkpoints/sam_vit_h_4b8939.pth'
    sam_predictor = SamPredictor(sam_model_registry['vit_h'](checkpoint=sam_checkpoint).to(device))

    glip_thr = 0
    npys = [x.replace('.jpg', '.npy') for x in images]
    for one_path in tqdm(npys):
        image_path = os.path.join(image_dir, one_path.replace('.npy', '.jpg'))
        result_path = os.path.join(glip_res_dir, one_path.split('/')[-1])
        glip_gpt_result = np.load(result_path, allow_pickle=True).item()
        print(image_path)
        output = {}
        for one_cand in glip_gpt_result.keys():
            output[one_cand] = {}
            print(one_cand)
            if len(glip_gpt_result[one_cand]) == 0:
                output[one_cand]['grounded_bboxes'] = []
                output[one_cand]['grounded_scores'] = []
                output[one_cand]['grounded_phrase_masks'] = []
                output[one_cand]['grounded_phrases'] = []
                output[one_cand]['grounded_phrase_entities'] = []
                continue

            grounded_phrases = []
            grounded_bboxes = []
            grounded_scores = []
            grounded_phrase_entities = []
            grounded_phrase_masks = []
            for phrase_i in range(len(glip_gpt_result[one_cand])):
                phrase_result = glip_gpt_result[one_cand][phrase_i]
                phrase = phrase_result['caption']
                entities = phrase_result['entities']
                bboxes = phrase_result['bboxes'].tolist()
                labels = phrase_result['labels'].tolist()
                scores = phrase_result['scores'].tolist()
                #print("\nphrase:", phrase)

                phrase_bboxes = []
                phrase_etts = []
                phrase_masks = []
                phrase_scores = []
                #print(phrase_i, entities)
                for ett_i, one_ett in enumerate(entities):
                    ett_idx = ett_i + 1
                    ett_pos = [i for i in range(len(labels)) if labels[i]==ett_idx and scores[i]>glip_thr]
                    ett_bboxes = [bboxes[x] for x in ett_pos]
                    ett_scores = [scores[x] for x in ett_pos]
                    if ett_bboxes:
                        ett_mask = bboxes2masks(sam_predictor, image_path, torch.as_tensor(ett_bboxes), device).cpu().numpy()
                        phrase_bboxes.append(ett_bboxes)
                        phrase_masks.append(ett_mask)
                        phrase_etts.append(one_ett)
                        phrase_scores.append(ett_scores)

                #print('valid phrase', len(phrase_bboxes), phrase_etts)

                grounded_bboxes.append(phrase_bboxes)
                grounded_scores.append(phrase_scores)
                grounded_phrase_masks.append(phrase_masks)
                grounded_phrases.append(phrase)
                grounded_phrase_entities.append(phrase_etts)

                #ipdb.set_trace()
            output[one_cand]['grounded_bboxes'] = grounded_bboxes
            output[one_cand]['grounded_scores'] = grounded_scores
            output[one_cand]['grounded_phrase_masks'] = grounded_phrase_masks
            output[one_cand]['grounded_phrases'] = grounded_phrases
            output[one_cand]['grounded_phrase_entities'] = grounded_phrase_entities
        np.save(os.path.join(output_dir, one_path.split('/')[-1].replace('.npy', '_glip_with_sam.npy')), output)

if __name__ == '__main__':
    glip_res_dir = '../HPG_example/glip_res'
    image_dir = '../HPG_example/images'
    output_dir = '../HPG_example'

    images = os.listdir(image_dir)

    print(len(images))
    glip_with_sam(images, image_dir, output_dir)