#%%
import numpy as np
import os
from tqdm import tqdm

from torch.nn import functional as F
from mask_utils import get_image_parsing_graph, bboxes2masks, convert_oneformer_format_to_sam, merge_sam_and_oneformer, merge_glip_mask_to_img_all_masks, mask_nms, build_tree_with_root_glip, get_segidx2bbox_from_masks, show_anns, build_tree_with_root_glip_print
from glip_util_for_one_img import load_glip_with_sam
#%%
image_path = './images/2409524.jpg'
image_name = image_path.split('/')[-1]
image_prefix = image_path.split('/')[-1].split('.')[0]

glip_mask_dir = '../HPG_example'
sam_mask_dir = '../HPG_example'
oneformer_mask_dir = '../HPG_example'

glip_masks = np.load(os.path.join(glip_mask_dir, f'{image_prefix}_glip_with_sam.npy'), allow_pickle=True).item()
glip_masks_new = {}
for k, v in glip_masks.items():
    glip_masks_new[' '.join(k.split())] = v 
#%%
one_cand = list(glip_masks_new.keys())[0]
glip_masks = glip_masks_new[one_cand]


sam_masks = np.load(os.path.join(sam_mask_dir, f'{image_prefix}_sam.npy'), allow_pickle=True).item()[image_name]
oneformer_masks = np.load(os.path.join(oneformer_mask_dir, f'{image_prefix}_oneformer.npy'), allow_pickle=True).item()[image_prefix]
oneformer_masks = convert_oneformer_format_to_sam(oneformer_masks) 
all_masks = merge_sam_and_oneformer(sam_masks, oneformer_masks, same_threshold=0.85)

filtered_masks, keep_indices = mask_nms(all_masks, threshold=0.9)

filtered_masks_with_glip = merge_glip_mask_to_img_all_masks(glip_masks, filtered_masks, same_threshold=0.85)

grounded_phrases, grounded_phrase_idxs, grounded_union_bboxes, grounded_all_bboxes = load_glip_with_sam(glip_masks)

segidx2bbox = get_segidx2bbox_from_masks(filtered_masks_with_glip)
grounded_seg_idxs = []
for grounded_i in range(len(grounded_phrases)):
    one_all_bboxes = grounded_all_bboxes[grounded_i]
    one_grounded_seg_idxs = []
    for one in one_all_bboxes:
        for x in segidx2bbox:
            if one in segidx2bbox[x]:
                one_grounded_seg_idxs.append(x)
    grounded_seg_idxs.append(one_grounded_seg_idxs)

root_node = build_tree_with_root_glip(filtered_masks_with_glip, image_path, global_discard_threshold=0.03, discard_threshold=0.05, contain_threshold=0.75)
root_node.print_tree()
# %%
