import os
import json

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/color_FNs_GTid.json', 'r') as j:
    color_FNs = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/IR_FN_GTids.json', 'r') as j:
    IR_FNs = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/fusion_fns_gtid.json', 'r') as j:
    fusion_FNs = json.load(j)

# import pdb;pdb.set_trace()

remained_FNs = list((set(color_FNs) & set(IR_FNs)) & set(fusion_FNs)) 
remained_FNs = sorted(remained_FNs)

with open('./remained_FNs_edit.json', 'w') as j:
    json.dump(remained_FNs, j)