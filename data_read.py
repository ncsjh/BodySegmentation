import os
import json
from tqdm import tqdm

def txtWrite(text):
    with open('./input_data/class_label.txt', 'a') as f:
        for i in text:
            f.write(f"{i}\n")
    f.close()
    return 0


# path='/home/inviz/Data/label'
# dirs=os.listdir(path)
#lbs=[]
# for dir in tqdm(dirs,desc='첫번째 for문'):
#     if len(dir)<10:
#         files = os.listdir(os.path.join(path, dir,'json'))
#         for file in files:
#             j_path=os.path.join(path, dir,'json', file)
#             try:
#                 a_j=open(j_path, encoding='utf-8')
#                 js=json.load(a_j)
#
#             except Exception as e:
#                 print(e, file)
#             for labels in js['labelingInfo']:
#                 if not labels['polygon']['label'] in lbs:
#                     lbs.append(labels['polygon']['label'])

#txtWrite(lbs)

def testSetWrite(target):
    org_list = ['train_id.txt','val_id.txt','test_id.txt']
    chan_list = ['train_tst_id.txt','val_tst_id.txt','test_test_id.txt']

    for idx,org in enumerate(org_list):
        #print(org)
        #print(f'./input_catalog/{org}')
        #print(f'./input_catalog/{chan_list[idx]}')

        with open(os.path.join('./input_catalog/',org),'r') as f:
            if os.path.isfile(os.path.join('./input_catalog',chan_list[idx])):
                #삭제
                os.remove(os.path.join('./input_catalog',chan_list[idx]))

            while True:
                line = f.readline()

                if not line:
                    break

                if str(target) in line:
                    with open(os.path.join('./input_catalog',chan_list[idx]),'a') as w:
                        w.write(line)

testSetWrite(500)
