import os
import json

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/color_FP_ids.json', 'r') as j:
    color_FPs = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/IR_fps_ids.json', 'r') as j:
    IR_FPs = json.load(j)

with open('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/fusion_fps_ids.json', 'r') as j:
    fusion_FPs = json.load(j)


new_FPs = list((set(fusion_FPs) - (set(color_FPs) | set(IR_FPs))))
new_FPs = sorted(new_FPs)

with open('./new_FPs.json', 'w') as j:
    json.dump(new_FPs, j)