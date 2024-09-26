import json
from score import print_score

dataset = 'flickr8k_exp'
glip_res_path = 'blip_matching_results/flickr8k_exp_modeltype-vg_dense_embsize-256_globimgfeat-True_splitcand-False_part0.json'

with open(glip_res_path, 'r') as f:
    res_data = json.load(f)
res_data = {int(k):v for k,v in res_data.items()}
for idx in res_data:
    res_data[idx]['id2info'] = {int(k):v for k,v in res_data[idx]['id2info'].items()}

print(dataset)
print_score(res_data, dataset)