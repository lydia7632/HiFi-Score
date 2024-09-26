import torch
import scipy.stats
import numpy as np
import ipdb
import copy

def repeat_3_times(scores):
    return [x for x in scores for _ in range(3)]

def calculate_f1_score(score_a, score_b):
    return (2 * score_a * score_b) / (score_a + score_b) if (score_a + score_b) != 0 else 0.0

def calculate_fbeta_score(score_a, score_b, beta):
    return (1+beta**2) * score_a * score_b / (beta**2 * score_a + score_b) if (score_a + score_b) != 0 else 0.0


def calculate_union_saliency(saliency_map, masks):
    union_mask = np.logical_or.reduce(masks)
    union_saliency_map = np.where(union_mask, saliency_map, 0)
    union_saliency = np.sum(union_saliency_map) / np.sum(saliency_map)
    
    return union_saliency


def calculate_union_area_ratio(masks):
    union_mask = np.logical_or.reduce(masks)
    union_area = np.sum(union_mask)
    total_area = masks[0].shape[0] * masks[0].shape[1]
    area_ratio = union_area / total_area
    
    return area_ratio

def calculate_union_saliency_with_sim(saliency_map, masks, sim_scores):

    union_mask = np.any(masks, axis=0)

    pixel_confidences = np.zeros_like(saliency_map)
    for i, (mask, confidence) in enumerate(zip(masks, sim_scores)):
        pixel_confidences += mask * confidence 

    mask_counts = np.sum(masks, axis=0)
    mask_counts[mask_counts == 0] = 1 

    average_confidences = pixel_confidences / mask_counts

    combined_saliency = saliency_map * average_confidences * union_mask

    overall_saliency = np.sum(combined_saliency) / np.sum(saliency_map)

    return overall_saliency

def calculate_union_with_sim(masks, sim_scores):

    union_mask = np.any(masks, axis=0)

    pixel_confidences = np.zeros((masks[0].shape[0], masks[0].shape[1]))
    for i, (mask, confidence) in enumerate(zip(masks, sim_scores)):
        pixel_confidences += mask * confidence  


    mask_counts = np.sum(masks, axis=0)
    mask_counts[mask_counts == 0] = 1 

    average_confidences = pixel_confidences / mask_counts  # better than next row

    return np.sum(average_confidences * union_mask)

def calculate_tree_t2i(global_score, similarities, phrase_ids, alpha=0.5):
    phrase_ids = torch.tensor(phrase_ids)
    similarities = torch.tensor(similarities)
    unique_ids, counts = phrase_ids.unique(return_counts=True)
    sums = torch.zeros_like(unique_ids).float()
    for id, count in zip(unique_ids, counts):
        if count > 1:
            sums[id] = alpha * similarities[phrase_ids == id][0] + (1-alpha) * similarities[phrase_ids == id][1:].sum()/(count - 1)
        elif count == 1:
            sums[id] = similarities[phrase_ids == id][0]
        else:
            sums[id] = 0

    t2i_score_tree_with_root = alpha * global_score + (1-alpha) * sums.mean()

    return t2i_score_tree_with_root

def add_sim_to_id2info(similarities, grounded_seg_idxs, id2info):
        segid2sim = {}
        for phrase_i in range(len(grounded_seg_idxs)):
            for one in grounded_seg_idxs[phrase_i]:
                if one not in segid2sim:
                    segid2sim[one] = []
                segid2sim[one].append(similarities[phrase_i+1])
        
        for one in id2info:
            id2info[one]['sim_score'] = 0
        id2info[-1]['sim_score'] = similarities[0]
        

        for one in segid2sim:
            if one in id2info:
                id2info[one]['sim_score'] = sum(segid2sim[one]) / len(segid2sim[one])
        #ipdb.set_trace()
        return id2info

def calculate_total_confidence(id2info, node_id, alpha=0.5, current_depth=0, max_depth=10, with_root=True):

    node = id2info[node_id]
    if current_depth >= max_depth:
        return node['sim_score']
    
    if not node['children']:
        return node['sim_score']
    
    children_confidence_sum = sum(calculate_total_confidence(id2info, child_id, alpha, current_depth=node['depth'], max_depth=max_depth) for child_id in node['children'])
    if node['children']:
        average_confidence = children_confidence_sum / len(node['children'])
    else:
        average_confidence = 0
    
    if with_root == False and current_depth==0:
        return average_confidence
    else:
        return alpha * node['sim_score'] + (1-alpha) * average_confidence


def calculate_total_confidence_saliency(id2info, node_id, alpha=0.5, current_depth=0, max_depth=10, with_root=True):

    node = id2info[node_id]
    #print(node['saliency_weight'], node['children'], current_depth, node['depth'])
    if current_depth >= max_depth:
        return node['saliency_weight'] * node['sim_score']
    
    if not node['children']:
        return node['saliency_weight'] * node['sim_score']
    children_confidence_sum = sum(calculate_total_confidence_saliency(id2info, child_id, alpha, current_depth=node['depth'], max_depth=max_depth) for child_id in node['children'])
    #print(children_confidence_sum)
    if node['children']:
        average_confidence = children_confidence_sum / len(node['children'])
    else:
        average_confidence = 0

    if with_root == False and current_depth==0:
        return average_confidence
    else:
        return node['saliency_weight'] * (alpha * node['sim_score'] + (1-alpha) * average_confidence)


def cal_t2i_sim_score(similarities, phrase_ids):

    if len(similarities) == 0:
        t2i_score = 0
        t2i_score_key = 0
        t2i_score_tree = 0
        return t2i_score, t2i_score_key, t2i_score_tree

    #t2i_score = torch.mean(torch.cat((similarities, torch.zeros(len(empty_pos), device=device)))).cpu().numpy()
    t2i_score = sum(similarities) / len(similarities)

    first_occurrences = {x: phrase_ids.index(x) for x in set(phrase_ids)}
    key_pos = [v for k,v in first_occurrences.items()]
    similarities_key = [similarities[x] for x in range(len(similarities)) if x in key_pos]
    t2i_score_key = sum(similarities_key) / len(similarities_key)

    phrase_ids = torch.tensor(phrase_ids)
    similarities = torch.tensor(similarities)
    unique_ids, counts = phrase_ids.unique(return_counts=True)
    sums = torch.zeros_like(unique_ids).float()
    for id in unique_ids:
        sums[id] = similarities[phrase_ids == id].sum()
    averages = sums / counts.float()
    t2i_score_tree = averages.mean().tolist()

    return t2i_score, t2i_score_key, t2i_score_tree

def print_score(res_data, dataset):
    itc_scores, itm_scores, itc_itm_scores, human_scores, old_human_scores, categories = calculate_scores(res_data)
    if dataset == 'pascal50s':
        human_labels = human_scores[::2]
        categories = categories[::2]
        print('\nitc')
        for score_type in itc_scores:
            print(f'{score_type}:', end='')
            calculate_pairwise_ranking(itc_scores[score_type], categories, human_labels)
        print('\nitm')
        for score_type in itm_scores:
            print(f'{score_type}:', end='')
            calculate_pairwise_ranking(itm_scores[score_type], categories, human_labels)
        print('\nitc+itm')
        for score_type in itc_itm_scores:
            print(f'{score_type}:', end='')
            calculate_pairwise_ranking(itc_itm_scores[score_type], categories, human_labels)

    elif dataset == 'flickr8k_exp':
        tauvariant = 'c'
        old_itc_scores = {score_type:repeat_3_times(itc_scores[score_type]) for score_type in itc_scores}
        old_itm_scores = {score_type:repeat_3_times(itm_scores[score_type]) for score_type in itm_scores}
        old_itc_itm_scores = {score_type:repeat_3_times(itc_itm_scores[score_type]) for score_type in itc_itm_scores}
        calculate_correlation(old_itc_scores, old_itm_scores, old_itc_itm_scores, old_human_scores, tauvariant)
    
    elif 'img_par' in dataset:
        print('\nitc')
        #ipdb.set_trace()
        for score_type in itc_scores:
            print(score_type, '{:.3f}'.format(calculate_pairwise(itc_scores[score_type])))
        print('\nitm')
        for score_type in itm_scores:
            print(score_type, '{:.3f}'.format(calculate_pairwise(itm_scores[score_type])))

        print('\nitc+itm')
        for score_type in itc_itm_scores:
            print(score_type, '{:.3f}'.format(calculate_pairwise(itc_itm_scores[score_type])))

    elif dataset == 'thumb':
        calculate_pearson_correlation(itc_scores, itm_scores, itc_itm_scores, human_scores)

    else:
        tauvariant = 'c'
        calculate_correlation(itc_scores, itm_scores, itc_itm_scores, human_scores, tauvariant)
    return itc_scores, itm_scores, itc_itm_scores
    
def calculate_scores(res_data, gamma = 1):
    score_types = {'add':[]}

    itc_scores = copy.deepcopy(score_types)
    itm_scores = copy.deepcopy(score_types)
    itc_itm_scores = copy.deepcopy(score_types)

    human_scores = []
    old_human_scores = []
    categories = []

    # print(len(res_data))
    #ipdb.set_trace()
    for idx in range(len(res_data)):
        if isinstance(res_data[idx]['human_score'], list):
            human_scores.append(sum(res_data[idx]['human_score']) / len(res_data[idx]['human_score']))
            old_human_scores += res_data[idx]['human_score']
        else:
            human_scores.append(res_data[idx]['human_score'])
        if res_data[idx]['category']:
            categories.append(res_data[idx]['category'])

        itm_similarities = res_data[idx]['itm']
        itc_similarities = res_data[idx]['itc']
        phrase_ids = res_data[idx]['phrase_ids']
        grounded_phrase_idxs = res_data[idx]['grounded_phrase_idxs']
        grounded_seg_idxs = res_data[idx]['grounded_seg_idxs']
        id2info = res_data[idx]['id2info']
        saliency_weights = res_data[idx]['saliency_weights']
        saliency_weights = np.nan_to_num(saliency_weights)

        empty_pos = [x for x in range(len(phrase_ids)) if  x not in grounded_phrase_idxs]
        t2i_itm_similarities = itm_similarities[1:].copy()
        t2i_itc_similarities = itc_similarities[1:].copy()
        #empty_pos = [x+1 for x in empty_pos]
        for index in sorted(empty_pos):
            t2i_itm_similarities.insert(index, 0)
            t2i_itc_similarities.insert(index, 0)

        itc_global_score = itc_similarities[0]
        itm_global_score = itm_similarities[0]

        itc_t2i_score_tree_new_with_root = calculate_tree_t2i(itc_global_score, t2i_itc_similarities, phrase_ids, alpha=0.5)
        itm_t2i_score_tree_new_with_root = calculate_tree_t2i(itm_global_score, t2i_itm_similarities, phrase_ids, alpha=0.5)


        segid2itmsim = {}
        segid2itcsim = {}
        for phrase_i in range(len(grounded_seg_idxs)):
            if len(grounded_seg_idxs[phrase_i])==1:
                if grounded_seg_idxs[phrase_i][0] not in segid2itmsim:
                    segid2itmsim[grounded_seg_idxs[phrase_i][0]] = 0
                    segid2itcsim[grounded_seg_idxs[phrase_i][0]] = 0
                segid2itmsim[grounded_seg_idxs[phrase_i][0]] = max(segid2itmsim[grounded_seg_idxs[phrase_i][0]], itm_similarities[phrase_i+1])
                segid2itcsim[grounded_seg_idxs[phrase_i][0]] = max(segid2itcsim[grounded_seg_idxs[phrase_i][0]], itc_similarities[phrase_i+1])


        for one in id2info:
            if one!=-1:
                id2info[one]['saliency_weight'] = saliency_weights[int(one)]
            else: 
                id2info[one]['saliency_weight'] = 1
        #ipdb.set_trace()
        alpha = 0.5
        max_depth = max([id2info[x]['depth'] for x in id2info])

        #print('max_depth', max_depth)
        id2info = add_sim_to_id2info(itm_similarities, grounded_seg_idxs, id2info)
        itm_i2t_score_tree_saliency_with_root = calculate_total_confidence_saliency(id2info, -1, alpha = alpha, current_depth=0, max_depth=max_depth, with_root=True)

        id2info = add_sim_to_id2info(itc_similarities, grounded_seg_idxs, id2info)
        itc_i2t_score_tree_saliency_with_root = calculate_total_confidence_saliency(id2info, -1, alpha = alpha, current_depth=0, max_depth=max_depth, with_root=True)

        itc_scores['add'].append(itc_t2i_score_tree_new_with_root + gamma * itc_i2t_score_tree_saliency_with_root)
        itm_scores['add'].append(itm_t2i_score_tree_new_with_root + gamma * itm_i2t_score_tree_saliency_with_root)

        itc_itm_t2i_score_tree_new_with_root = itc_t2i_score_tree_new_with_root + itm_t2i_score_tree_new_with_root
        itc_itm_i2t_score_tree_saliency_with_root = itc_i2t_score_tree_saliency_with_root + itm_i2t_score_tree_saliency_with_root
        itc_itm_scores['add'].append(itc_itm_t2i_score_tree_new_with_root + gamma * itc_itm_i2t_score_tree_saliency_with_root)


    return itc_scores, itm_scores, itc_itm_scores, human_scores, old_human_scores, categories


def calculate_correlation(itc_scores, itm_scores, itc_itm_scores, human_scores, tauvariant):
    tauvariant = 'c'
    print('\nitc')
    for score_type in itc_scores:
        print('Tau-{}: {:.3f}'.format(tauvariant, 100*scipy.stats.kendalltau(itc_scores[score_type], human_scores, variant=tauvariant)[0]))
    
    print('\nitm')
    for score_type in itm_scores:
        print('Tau-{}: {:.3f}'.format(tauvariant, 100*scipy.stats.kendalltau(itm_scores[score_type], human_scores, variant=tauvariant)[0]))

    print('\nitc+itm')
    for score_type in itc_itm_scores:
        print('Tau-{}: {:.3f}'.format(tauvariant, 100*scipy.stats.kendalltau(itc_itm_scores[score_type], human_scores, variant=tauvariant)[0]))


def calculate_pearson_correlation(itc_scores, itm_scores, itc_itm_scores, human_scores):
    #ipdb.set_trace()
    P_human_scores = [x['P'] for x in human_scores]
    R_human_scores = [x['R'] for x in human_scores]
    total_human_scores = [x['human_score'] for x in human_scores]
    print('\nitc')
    for score_type in itc_scores:
        print(score_type, 'pearsonr: {:.3f} {:.3f} {:.3f}'.format(100*scipy.stats.pearsonr(itc_scores[score_type], P_human_scores)[0], 100*scipy.stats.pearsonr(itc_scores[score_type], R_human_scores)[0], 100*scipy.stats.pearsonr(itc_scores[score_type], total_human_scores)[0]))
    #ipdb.set_trace()

    print('\nitm')
    for score_type in itm_scores:
        print(score_type, 'pearsonr: {:.3f} {:.3f} {:.3f}'.format(100*scipy.stats.pearsonr(itm_scores[score_type], P_human_scores)[0], 100*scipy.stats.pearsonr(itm_scores[score_type], R_human_scores)[0], 100*scipy.stats.pearsonr(itm_scores[score_type], total_human_scores)[0]))

    print('\nitc+itm')
    for score_type in itc_itm_scores:
        print(score_type, 'pearsonr: {:.3f} {:.3f} {:.3f}'.format(100*scipy.stats.pearsonr(itc_itm_scores[score_type], P_human_scores)[0], 100*scipy.stats.pearsonr(itc_itm_scores[score_type], R_human_scores)[0], 100*scipy.stats.pearsonr(itc_itm_scores[score_type], total_human_scores)[0]))


def calculate_pairwise(scores):
    correct_samples = 0
    all_samples = 0
    for i in range(0, len(scores), 2):
        gt_score = scores[i]
        score_1 = scores[i+1]
        if gt_score > score_1:
            correct_samples += 1
        all_samples += 1
    #print(f"{correct_samples}/{all_samples}={correct_samples/all_samples}")
    return 100*correct_samples/all_samples