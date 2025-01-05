#%%
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import shutil
import pydot
import os
import cv2
import inflect
p = inflect.engine()


background_classes = ['door-stuff', 'floor-wood', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']


def bboxes2masks(sam_predictor, image_path, bboxes, device):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image)
    # bboxes = torch.as_tensor([[ 25.5320,  25.3889, 483.8209, 319.6260]])
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(bboxes, image.shape[:2]).to(device)
    masks, _, _ = sam_predictor.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device),
        multimask_output = False,
    )
    return masks

def show_anns(image_path, anns, labels=None):

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if len(anns) == 0:
        return
    if labels:
        assert len(anns) == len(labels), "Length of annotations and labels should be the same."
    else:
        labels = list(range(len(anns)))

    #sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)

    sorted_data = sorted(zip(anns, labels), key=lambda x: x[0]['area'], reverse=True)
    sorted_anns = [item[0] for item in sorted_data]
    sorted_labels = [item[1] for item in sorted_data]
    #print("sorted_anns", sorted_anns)
    #print("sorted_labels", sorted_labels)

    plt.figure(figsize=(20,20))
    plt.imshow(image)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((image.shape[0], image.shape[1], 4))
    img[:,:,3] = 0
    
    #for index, (ann, label) in enumerate(zip(anns, labels)):
    for index, (ann, label) in enumerate(zip(sorted_anns, sorted_labels)):
        #print(ann['bbox'], ann['area'])
        if 'segmentation' in ann:
            m = ann['segmentation']
            #color_mask = np.concatenate([np.random.random(3), [0.35]])
            color_mask = np.concatenate([np.random.random(3), [0.8]])
            img[m] = color_mask

        # Draw the bounding box
        bbox = ann['bbox'] # xywh
        rect = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)

        ax.text(bbox[0], bbox[1], str(label), color='white', bbox=dict(facecolor='red', edgecolor='red', boxstyle='round,pad=0.2'))

    ax.imshow(img)
    
    plt.axis('off')
    plt.show()


def draw_bboxes(im_file, boxes, labels=None, inds=None):
    plt.figure()
    im = cv2.imread(im_file)

    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(im)

    if not inds:
        inds = list(range(len(boxes)))
    for i in inds:
        bbox = boxes[i]
        if bbox[0] == 0:
            bbox[0] = 1
        if bbox[1] == 0:
            bbox[1] = 1

        plt.gca().add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                        bbox[2] - bbox[0],
                        bbox[3] - bbox[1], fill=False,
                        edgecolor='red', linewidth=2, alpha=0.5)
                )
        if labels:
            plt.gca().text(bbox[0], bbox[1] - 2,
                        '%s' % (labels[i]),
                        bbox=dict(facecolor='blue', alpha=0.5),
                        fontsize=10, color='white')

def segment_image(image, segmentation_mask):
    image_array = np.array(image)
    segmented_image_array = np.zeros_like(image_array)
    segmented_image_array[segmentation_mask] = image_array[segmentation_mask]
    segmented_image = Image.fromarray(segmented_image_array)
    black_image = Image.new("RGB", image.size, (255, 255, 255))
    #black_image = Image.new("RGB", image.size, (0, 0, 0))
    transparency_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
    transparency_mask[segmentation_mask] = 255
    transparency_mask_image = Image.fromarray(transparency_mask, mode='L')
    black_image.paste(segmented_image, mask=transparency_mask_image)
    return black_image


class TreeNode:
    def __init__(self, data, image_path=None, index=None):
        self.data = data
        self.index = index  # 在segments中的顺序标号
        self.children = []
        self.image_path = image_path
        self.depth = 0

    def contains_index(self, index):
        # 首先检查当前节点的index
        if self.index == index:
            return True
        # 然后遍历所有子节点
        for child in self.children:
            if child.contains_index(index):
                return True
        # 如果当前节点和所有子节点都不匹配，返回False
        return False

    def add_child(self, node):
        self.children.append(node)
        self.change_depth()

    def change_depth(self):
        for child in self.children:
            child.depth = self.depth + 1
            child.change_depth()

    def print_tree(self):
        # 如果节点有数据，打印数据的相关信息，否则打印'Root'
        if self.data:
            child_indices = [child.index for child in self.children]
            print('  ' * self.depth + f"Segment #{self.index} (Area: {self.data['area']}) depth={self.depth}")
            #print("?", child_indices)
            #if child_indices:
            #    print('  ' * (indent+1) + f'{child_indices}')
        else:
            print('  ' * self.depth + "Root")

        for child in self.children:
            child.print_tree()


    def pydot_print_tree(self, graph, GraphParentNode=None, max_depth=100):
        image_prefix = self.image_path.split('/')[-1].split('.')[0]
        if self.depth <= max_depth:
            if self.depth>0:            
                child_indices = [child.index for child in self.children]
                print('  ' * self.depth + f"Segment #{self.index} (Area: {self.data['area']})  depth={self.depth}")
                #print("?", self.index, child_indices)
                #current_graph_node = pydot.Node(f"Child {self.index}", label=f"Child {self.index}", shape='none')
                node_label =  f"Child {self.index} - {self.data['category']}" if 'category' in self.data else f"Child {self.index}"
                current_graph_node = pydot.Node(f"Child {self.index}", label=node_label, shape='none', image=f"/home/lm1/projects/HSG/cap_eval_with_grounding/camera-ready/{image_prefix}/{self.index}.jpg", labelloc="t")
                graph.add_node(current_graph_node)
                edge = pydot.Edge(GraphParentNode, current_graph_node)
                graph.add_edge(edge)
                #if child_indices:
                #    print('  ' * (indent+1) + f'{child_indices}')
            else:
                print('  ' * self.depth + "Root")
                current_graph_node = pydot.Node("Root", label="Root Node", shape='none', image=self.image_path)
                graph.add_node(current_graph_node)

            # if self.data and 'category' in self.data and self.data['category'] in background_classes:
            #     pass
            # else:
            #     for child in self.children:
            #         child.pydot_print_tree(graph, GraphParentNode=current_graph_node)
            for child in self.children:
                child.pydot_print_tree(graph, GraphParentNode=current_graph_node)

    def walking_through_tree(self, parent_index, id2info):
        current_area = int(self.data['area']) if self.data else None
        id2info[self.index] = {'depth':self.depth, 'children':[x.index for x in self.children], 'parent':parent_index, 'area':current_area}
        for child in self.children:
            child.walking_through_tree(self.index, id2info)


def is_contained(segmentation_A, segmentation_B):
    return np.all(np.logical_or(np.logical_not(segmentation_A), segmentation_B))

def is_mostly_contained(segmentation_A, segmentation_B, threshold=0.85):
    overlap = np.logical_and(segmentation_A, segmentation_B)
    overlap_ratio = np.sum(overlap) / np.sum(segmentation_A)    
    return overlap_ratio >= threshold

def overlap(segmentation_A, segmentation_B):
    overlap = np.logical_and(segmentation_A, segmentation_B)
    overlap_ratio = np.sum(overlap) / np.sum(segmentation_A)   
    return overlap_ratio

def iou(segmentation_A, segmentation_B):
    overlap = np.logical_and(segmentation_A, segmentation_B)
    union = np.logical_or(segmentation_A, segmentation_B)
    iou = np.sum(overlap) / np.sum(union)    
    return iou

def is_same(segmentation_A, segmentation_B, threshold=0.9):
    overlap = np.logical_and(segmentation_A, segmentation_B)
    union = np.logical_or(segmentation_A, segmentation_B)
    iou = np.sum(overlap) / np.sum(union)    
    return iou >= threshold

def build_tree_with_root(segments, image_path, discard_threshold=0.1, contain_threshold=0.92):
    sorted_indices = sorted(range(len(segments)), key=lambda k: segments[k]['area'])
    nodes = [TreeNode(segments[x], image_path=image_path, index=x) for x in sorted_indices]
    root = TreeNode(None, image_path=image_path, index=-1)
    root_area = segments[0]['segmentation'].shape[0] * segments[0]['segmentation'].shape[1]
    
    for i in range(len(segments)):
        is_child_of_any = False
        for j in range(i + 1, len(segments)):
            #print(f"check if {i} to {j}")
            if is_mostly_contained(segments[sorted_indices[i]]['segmentation'], segments[sorted_indices[j]]['segmentation'], threshold=contain_threshold):
                is_child_of_any = True
                if segments[sorted_indices[i]]['area'] / segments[sorted_indices[j]]['area'] > discard_threshold:
                    nodes[j].add_child(nodes[i])
                #    print(f"add {sorted_indices[i]} to {sorted_indices[j]}")
                # else:
                #    print(sorted_indices[i], 'too small')
                break 

        if not is_child_of_any:
            if segments[sorted_indices[i]]['area'] / root_area > discard_threshold:
                root.add_child(nodes[i])
            #    print(f"add {sorted_indices[i]} to root")
            #else:
            #    print(sorted_indices[i], 'too small, not add to root')
    return root

def build_tree_with_root_glip(segments, image_path, global_discard_threshold=0.1, discard_threshold=0.1, contain_threshold=0.92):
    sorted_indices = sorted(range(len(segments)), key=lambda k: segments[k]['area'])
    nodes = [TreeNode(segments[x], image_path=image_path, index=x) for x in sorted_indices]
    root = TreeNode(None, image_path=image_path, index=-1)  # 创建根节点
    root_area = segments[0]['segmentation'].shape[0] * segments[0]['segmentation'].shape[1]
    
    #ipdb.set_trace()
    for i in range(len(segments)):
        # if sorted_indices[i] == 100:
        #     print("try to find parent")
        is_child_of_any = False
        for j in range(i + 1, len(segments)):
            #print(f"check if {i} to {j}")
            if is_mostly_contained(segments[sorted_indices[i]]['segmentation'], segments[sorted_indices[j]]['segmentation'], threshold=contain_threshold):
                is_child_of_any = True
                if (segments[sorted_indices[i]]['area'] / root_area > global_discard_threshold and segments[sorted_indices[i]]['area'] / segments[sorted_indices[j]]['area'] > discard_threshold) or segments[sorted_indices[i]]['glip_grounded_bboxes']:
                    nodes[j].add_child(nodes[i])
                #     print(f"add {sorted_indices[i]} to {sorted_indices[j]}")
                # else:
                #    print(sorted_indices[i], 'too small')
                break

        if not is_child_of_any:
            # if sorted_indices[i] == 100:
            #     print("add to  root")
            if (segments[sorted_indices[i]]['area'] / root_area > global_discard_threshold and segments[sorted_indices[i]]['area'] / root_area > discard_threshold) or segments[sorted_indices[i]]['glip_grounded_bboxes']:
                root.add_child(nodes[i])
            #    print(f"add {sorted_indices[i]} to root")
            # else:
            #    print(sorted_indices[i], 'too small, not add to root')
    return root


def build_tree_with_root_glip_print(segments, image_path, global_discard_threshold=0.1, discard_threshold=0.1, contain_threshold=0.92):
    sorted_indices = sorted(range(len(segments)), key=lambda k: segments[k]['area'])
    nodes = [TreeNode(segments[x], image_path=image_path, index=x) for x in sorted_indices]
    root = TreeNode(None, image_path=image_path, index=-1)  # 创建根节点
    root_area = segments[0]['segmentation'].shape[0] * segments[0]['segmentation'].shape[1]

    #ipdb.set_trace()
    for i in range(len(segments)):
        # if sorted_indices[i] == 100:
        #     print("try to find parent")
        is_child_of_any = False
        for j in range(i + 1, len(segments)): 
            #print(f"check if {i} to {j}")
            if is_mostly_contained(segments[sorted_indices[i]]['segmentation'], segments[sorted_indices[j]]['segmentation'], threshold=contain_threshold):
                is_child_of_any = True
                if (segments[sorted_indices[i]]['area'] / root_area > global_discard_threshold and segments[sorted_indices[i]]['area'] / segments[sorted_indices[j]]['area'] > discard_threshold) or segments[sorted_indices[i]]['glip_grounded_bboxes']:
                    nodes[j].add_child(nodes[i])
                    print(f"add {sorted_indices[i]} to {sorted_indices[j]}")
                else:
                   print(sorted_indices[i], 'too small')
                break 

        if not is_child_of_any:
            if (segments[sorted_indices[i]]['area'] / root_area > global_discard_threshold and segments[sorted_indices[i]]['area'] / root_area > discard_threshold) or segments[sorted_indices[i]]['glip_grounded_bboxes']:
                root.add_child(nodes[i])
                print(f"add {sorted_indices[i]} to root")
            else:
               print(sorted_indices[i], 'too small, not add to root')
    
    for i in range(len(segments)):
        if not root.contains_index(sorted_indices[i]) and segments[sorted_indices[i]]['glip_grounded_bboxes']:
            root.add_child(nodes[i])
            print(f"GLIP find it, add {sorted_indices[i]} to root")

    return root


def mask_nms(segments, threshold=0.9):
    keep_indices = []
    left_indices = sorted(range(len(segments)), key=lambda k: segments[k]['area'])

    while len(left_indices):
        current_idx = left_indices[0]
        current_segmentation = segments[left_indices[0]]['segmentation']
        keep_indices.append(left_indices[0])
        left_indices.remove(current_idx)

        compare_indices = left_indices.copy()

        for j in compare_indices:
            if is_same(current_segmentation, segments[j]['segmentation'], threshold=threshold):
                left_indices.remove(j)
    
    filtered_segments = [segments[x] for x in keep_indices]
    return filtered_segments, keep_indices

def find_bbox_from_mask(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    return [xmin, ymin, xmax-xmin, ymax-ymin]

def convert_oneformer_format_to_sam(oneformer_res):
    stuff_classes=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

    total_mask, cate = oneformer_res
    total_mask = total_mask.cpu().numpy()
    oneformer_masks = []
    for i, one in enumerate(cate):
        one_mask_id = one['id']
        one_mask = (total_mask==one_mask_id)
        one_area = np.sum(one_mask)
        one_bbox = find_bbox_from_mask(one_mask)
        one_cate = stuff_classes[one['category_id']]
        oneformer_masks.append({'segmentation':one_mask, 'area':one_area, 'bbox':one_bbox, 'category':stuff_classes[one['category_id']]})
    return oneformer_masks

def merge_sam_and_oneformer(sam_masks, oneformer_masks, same_threshold=0.85):
    oneformer_found_indices = []
    for i, one_sem_mask in enumerate(oneformer_masks):
        for j in range(len(sam_masks)):
            if is_same(one_sem_mask['segmentation'], sam_masks[j]['segmentation'], threshold=same_threshold):
                sam_masks[j] = one_sem_mask
                oneformer_found_indices.append(i)
    all_masks = sam_masks + [oneformer_masks[x] for x in range(len(oneformer_masks)) if x not in oneformer_found_indices]

    for i in range(len(all_masks)):
        all_masks[i]['glip_grounded_bboxes'] = []
    return all_masks

def merge_glip_mask_to_img_all_masks(glip_masks, all_masks, same_threshold=0.85):
    grounded_bboxes = glip_masks['grounded_bboxes']
    grounded_phrase_masks = glip_masks['grounded_phrase_masks']
    grounded_phrases = glip_masks['grounded_phrases']
    grounded_phrase_entities = glip_masks['grounded_phrase_entities']
    grounded_scores = glip_masks['grounded_scores']
    for phrase_i in range(len(grounded_phrases)):
        phrase = grounded_phrases[phrase_i]
        phrase_etts = grounded_phrase_entities[phrase_i]
        bboxes = grounded_bboxes[phrase_i]
        masks = grounded_phrase_masks[phrase_i]
        #print(phrase, phrase_etts, len(phrase_etts), len(bboxes), len(masks))

        for ett_i, one_ett in enumerate(phrase_etts):
            for one_bbox, one_mask in zip(bboxes[ett_i], masks[ett_i]):
                one_mask = one_mask.squeeze(0)
                find = 0
                for i in range(len(all_masks)):
                    if is_same(all_masks[i]['segmentation'], one_mask, threshold=same_threshold):
                        find = 1
                        all_masks[i]['glip_grounded_bboxes'].append(one_bbox)
                        break

                if find == 0:
                    all_masks.append({'segmentation':one_mask, 'area':one_mask.sum(), 'bbox':[one_bbox[0], one_bbox[1], one_bbox[2]-one_bbox[0], one_bbox[3]-one_bbox[1]], 'glip_grounded_bboxes':[one_bbox], 'source':'glip'})
    
    return all_masks

def get_segidx2bbox_from_masks(all_masks):
    segidx2bbox = {}
    for seg_i, one in enumerate(all_masks):
        if one['glip_grounded_bboxes']:
            segidx2bbox[seg_i] = one['glip_grounded_bboxes']

    return segidx2bbox


def get_image_parsing_graph(sam_masks, oneformer_masks, image_path):
    oneformer_masks = convert_oneformer_format_to_sam(oneformer_masks) 

    #all_masks = sam_masks + oneformer_masks
    all_masks = merge_sam_and_oneformer(sam_masks, oneformer_masks, same_threshold=0.85)

    print(f"sam: {len(sam_masks)}, oneformer: {len(oneformer_masks)}, all: {len(all_masks)}")
    filtered_masks, keep_indices = mask_nms(all_masks, threshold=0.9)
    print(f"nms {len(all_masks)} -> {len(filtered_masks)}")
    #show_anns(image_path, filtered_masks)

    root_node = build_tree_with_root(filtered_masks, image_path, discard_threshold=0.02, contain_threshold=0.75)
    root_node.print_tree()

    return all_masks, root_node

#%%

def vis_temp():
    image_path = '/data/coco/val2014/COCO_val2014_000000035807.jpg'
    image_name = image_path.split('/')[-1]
    image_prefix = image_path.split('/')[-1].split('.')[0]
    oneformer_res = np.load('/home/lm1/vis_by_blip2/COCO_val2014_000000035807_oneformer.npy', allow_pickle=True).item()[image_prefix]
    oneformer_masks = convert_oneformer_format_to_sam(oneformer_res) 

    sam_results = np.load(f'/home/lm1/vis_by_blip2/COCO_val2014_000000035807.jpg_sam_result.npy', allow_pickle=True)
    sam_masks = sam_results.item()[image_name]
    all_masks = merge_sam_and_oneformer(sam_masks, oneformer_masks, same_threshold=0.85)

    filtered_masks, keep_indices = mask_nms(all_masks, threshold=0.9)

    root_node = build_tree_with_root(filtered_masks, image_path, discard_threshold=0.02, contain_threshold=0.75)
    root_node.print_tree()

    raw_image = Image.open(image_path).convert("RGB")
    if os.path.exists(f'pydot_segs/{image_prefix}'):
        shutil.rmtree(f'pydot_segs/{image_prefix}')
    os.makedirs(f'pydot_segs/{image_prefix}')

    for seg_i in range(len(filtered_masks)):
        bbox = [filtered_masks[seg_i]['bbox'][0], filtered_masks[seg_i]['bbox'][1], filtered_masks[seg_i]['bbox'][0]+max(filtered_masks[seg_i]['bbox'][2], 1), filtered_masks[seg_i]['bbox'][1]+max(filtered_masks[seg_i]['bbox'][3], 1)]
        bbox_image = segment_image(raw_image, filtered_masks[seg_i]['segmentation']).crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        bbox_image.save(f'pydot_segs/{image_prefix}/{seg_i}.jpg')

    graph = pydot.Dot(graph_type='graph')
    root_node.pydot_print_tree(graph, image_prefix)

    graph.write_jpeg(f'pydot_segs/{image_prefix}_tree.jpg')
    print(f"pydot graph saved at pydot_segs/{image_prefix}_tree.jpg")

if __name__ == '__main__':
    vis_temp()

# %%
