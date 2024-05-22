
import torch
from pedalboard.io import AudioFile
import random
import json
from tqdm import tqdm
import numpy as np

checkpoint = torch.load('BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt')
def id_to_name(label_idx):
    json_path ="ontology/ontology.json"
    with open(json_path, 'r') as f:
        data = json.load(f)
    mapdata = {item["id"]: item["name"] for item in data}
    result = [mapdata[id] for id in label_idx if id in mapdata]
    if 'Silence'  in result:
        result.remove('Silence')
    if 'Music' in result:
        result.remove('Music')
    if 'Speech' in result:
        result.remove('Speech')
    if 'Background music' in result:
        result.remove('Background music') 
    return result


probs = np.load('./probs.npy')
probs = torch.from_numpy(probs)

json_path ="json_files/raw_30s_cleantags.json"
with open(json_path, 'r') as f:
    data = json.load(f)
result =[]

for i, (top5_label_prob, top5_label_idx) in enumerate(zip(*probs.topk(k=5))):
    top5_label = [checkpoint['label_dict'][label_idx.item()] for label_idx in top5_label_idx]
    label = id_to_name(top5_label)
    labelstr = ', '.join(label)
    entry = {
        "location": data[i]['location'],
        "duration":data[i]['duration'],
        "tags": data[i]['tags'] + ',' + labelstr,
    }
    print(entry)
    result.append(entry)

with open('json_files/mtg_tags.json', 'w') as f:
    json.dump(result, f, indent=4)

