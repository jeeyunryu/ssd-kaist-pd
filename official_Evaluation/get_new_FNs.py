import os
import json

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/color_tps_dtm_ids.json', 'r') as j:
    color_TP_gtids = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/IR_tps_dtm_ids.json', 'r') as j:
    IR_TP_gtids = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/fusion_fns_gtid.json', 'r') as j:
    fusion_FNs = json.load(j)


new_FNs = list((set(fusion_FNs) & (set(color_TP_gtids) | set(IR_TP_gtids))))
new_FNs = sorted(new_FNs)

with open('./new_FNs.json', 'w') as j:
    json.dump(new_FNs, j)