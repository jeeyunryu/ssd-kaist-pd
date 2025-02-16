# GT 아이디 리스트 가져오기
# TP 가져오기
# FN 뽑기
# 파일에 저장하기
import os
import json

with open(os.path.join('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/gts_ids.json'), 'r') as j:
    gts = json.load(j)

with open(os.path.join('/home/urp4/workspace/codes_for_kaist/official_Evaluation/bbox_ids/IR_tps_dtm_ids.json'), 'r') as j: 
    tpms = json.load(j)

fns = list(set(gts) - set(tpms))
# tmp = list(set(gts) & set(tpms))
# import pdb;pdb.set_trace()
with open(os.path.join('./bbox_ids/IR_FN_GTids.json'), 'w') as j:
    json.dump(fns, j)

