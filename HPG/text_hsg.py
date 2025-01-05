#%%
import json
import re
from pprint import pprint

clothing = ['bag', 'bags', 'backpack', 'blouse', 'boxing gloves', 'boxing shorts', 'boots', 'cap', 'dress', 'dresses', 'fedora', 'glasses', 'hardhat', 'hat', 'head scarves', 'headdress', 'helmet', 'helmets', 'hoodie', 'jacket', 'jackets', 'jeans', 'jersey', 'jerseys', 'leggings', 'outfit', 'safari hat', 'scarf', 'shirt', 'shirts', 'shorts', 'suit', 'suits', 'sunglasses', 'suspenders', 'sweater', 'swim trunks', 'swimming suit', 't-shirt', 'uniform', 'uniforms', 'mask', 'wetsuit', 'coat', 'coats', 'sweatshirt', 'sweater', 'sweaters']


def Textual_HSG(sent, gpt_result):
    MAX_DEP = 0
    count = 0

    objects = gpt_result['objects']
    for one_obj in objects:
        attr_str = re.sub(r'\(.*?\)', '', one_obj['Attributes'])
        one_obj['Attributes'] = [item.strip().lower() for item in attr_str.split(',')]
        one_obj['Attributes'] = [x for x in one_obj['Attributes'] if x != 'none']
        one_obj['Attributes'] = [x for x in one_obj['Attributes'] if x != 'one']
        one_obj['Attributes'] = [x for x in one_obj['Attributes'] if x != 'do']
        one_obj['Attributes'] = [x for x in one_obj['Attributes'] if x != 'take']
    print(objects)

    for item in objects:
        for key, value in item.items():
            if key != 'Attributes':
                item[key] = value.strip().lower()

    relations = gpt_result['relations']
    for sublist in relations:
        for i in range(len(sublist)):
            sublist[i] = sublist[i].strip().lower()


    # check objects with same name
    obj_list = [x['Object'] for x in objects]
    if len(obj_list) != len(set(obj_list)):
        print('\n', sent)
        print(obj_list)
        pprint(objects)
        print(relations)
        count += 1

    for one_object in objects:
        if 'none' in one_object['Object']:
            print("None!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(sent)
            print(one_obj)
        if one_object['Object'] == 'group' and 'group of' in sent.lower():
            print("group!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(sent)
            print(one_object)
        if '(' in one_object['Object'] or ')' in one_object['Object']:
            print("() in object!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(sent)
            print(one_object)       


    # check relations

    new_relations = []
    for one_relation in relations:
        if one_relation[0] not in obj_list:
            continue


        if one_relation[2]=='none':
            obj_idx = obj_list.index(one_relation[0])
            if one_relation[1] not in objects[obj_idx]['Attributes']:
                objects[obj_idx]['Attributes'].append(one_relation[1])
        elif one_relation[1] == 'do':
                obj_idx = obj_list.index(one_relation[0])
                if one_relation[2] not in objects[obj_idx]['Attributes']:
                    objects[obj_idx]['Attributes'].append(one_relation[2])
        else:
            new_relations.append(one_relation)


    # check PartOf
    parts = set([x['PartOf'] for x in objects])
    
    for one_obj in objects:
        if one_obj['PartOf'] != 'none':
            ett1 = one_obj['Object']
            ett2 = one_obj['PartOf']

            flag = 0
            related_rel = []
            for one_rel in relations:
                if ett1 in one_rel and ett2 in one_rel:
                    flag = 1
                    related_rel = one_rel.copy()

            if flag == 0 and ett1.split()[-1] in clothing:
                new_relations.append([ett2, 'wear', ett1])


    output = {'sent':sent, 'objects':[]}
    output_str = 'Description: {}\n'.format(sent)
    output_str += '\nObjects:\n'
    for idx, one_object in enumerate(objects):
        output_str += '{}. Object: {}\n'.format(idx+1, one_object['Object'])

        relations_needes = [x for x in new_relations if x[0]==one_object['Object']]
        attributes_needed = []
        for one_attr in one_object['Attributes']:
            attr_in_rel_flag = 0
            for one_rel in relations_needes:
                if one_attr in one_rel[1]:
                    attr_in_rel_flag = 1
            if attr_in_rel_flag == 0:
                attributes_needed.append(one_attr+' '+one_object['Object'])

        output['objects'].append({'object':one_object['Object'], 'attributes':attributes_needed, 'relations':relations_needes,})

    phrases = []
    phrase_ids = []
    for idx, one_obj in enumerate(output['objects']):
        phrases.append(one_obj['object'])
        phrase_ids.append(idx)
        for one_attr in one_obj['attributes']:
            phrases.append(one_attr)
            phrase_ids.append(idx)        
        for one_rel in one_obj['relations']:
            phrases.append(' '.join(one_rel))
            phrase_ids.append(idx)

    return output, phrases, phrase_ids


def Textual_HSG_new(sent, gpt_result):
    def extract_number(key):
        return int(key.split('_')[-1])

    sorted_keys = sorted(gpt_result, key=extract_number)

    def remove_number(value):
        return ' '.join([word.split('_')[0] for word in value.split()])

    phrases = []
    phrase_ids = []
    for idx, key in enumerate(sorted_keys):
        values = gpt_result[key]
        values = [remove_number(value) for value in values]
        key_without_num = key.split('_')[0]
        #print(f"{idx}, {key_without_num}: {values}")

        keep_values = [x for x in values if x != key_without_num and x != 'a '+key_without_num and x != 'the '+key_without_num]
        phrases.append(key_without_num)
        phrase_ids.append(idx)
        for one_phrase in keep_values:
            phrases.append(one_phrase)
            phrase_ids.append(idx) 
    return gpt_result, phrases, phrase_ids
