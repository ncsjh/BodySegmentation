# 1. path 설정
# 2. 데이터 전처리
# 3. 결과값 .txt 파이
import os
import threading
import tracemalloc

####지정
from threading import Thread

semaphore = threading.Semaphore(256)

from collections import OrderedDict
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)',text)]

def imageDataLoad(readPath):

    mCnt,fCnt = 0 , 0
    # 경로 지정
    resultFileName = "almost_full_label_data.txt"
    passDir1 = '/home/inviz/Data/Dataset/Label'
    passDir2 = '/home/inviz/Data/Dataset/Image'
    labellist = [[[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]]]
    imagelist = [[[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]],
                 [[],[],[],[],[],[]]]

    # :

    for (root, directories ,files) in os.walk(readPath):
        if passDir1 == root or passDir2 == root:
            pass
            #print('pass dir\t',root)
        else:
            if 'json' in root:
                if os.path.isdir(root):
                    list=os.listdir(root)
                    for file in list:
                        if file[:2]=='01': # Clothe ID가 01일때(ex : 01_02_F328_12)
                            if file[3:5] =='01':
                                labellist[0][0].append(file)
                            elif file[3:5] =='02':
                                labellist[0][1].append(file)
                            elif file[3:5] =='03':
                                labellist[0][2].append(file)
                            elif file[3:5] =='04':
                                labellist[0][3].append(file)
                            elif file[3:5] =='05':
                                labellist[0][4].append(file)
                            elif file[3:5] =='06':
                                labellist[0][5].append(file)
                        elif file[:2]=='02':
                            if file[3:5] =='01':
                                labellist[1][0].append(file)
                            elif file[3:5] =='02':
                                labellist[1][1].append(file)
                            elif file[3:5] =='03':
                                labellist[1][2].append(file)
                            elif file[3:5] =='04':
                                labellist[1][3].append(file)
                            elif file[3:5] =='05':
                                labellist[1][4].append(file)
                            elif file[3:5] =='06':
                                labellist[1][5].append(file)
                        elif file[:2]=='03':
                            if file[3:5] =='01':
                                labellist[2][0].append(file)
                            elif file[3:5] =='02':
                                labellist[2][1].append(file)
                            elif file[3:5] =='03':
                                labellist[2][2].append(file)
                            elif file[3:5] =='04':
                                labellist[2][3].append(file)
                            elif file[3:5] =='05':
                                labellist[2][4].append(file)
                            elif file[3:5] =='06':
                                labellist[2][5].append(file)
                        elif file[:2]=='04':
                            if file[3:5] =='01':
                                labellist[3][0].append(file)
                            elif file[3:5] =='02':
                                labellist[3][1].append(file)
                            elif file[3:5] =='03':
                                labellist[3][2].append(file)
                            elif file[3:5] =='04':
                                labellist[3][3].append(file)
                            elif file[3:5] =='05':
                                labellist[3][4].append(file)
                            elif file[3:5] =='06':
                                labellist[3][5].append(file)
                        elif file[:2]=='05':
                            if file[3:5] =='01':
                                labellist[4][0].append(file)
                            elif file[3:5] =='02':
                                labellist[4][1].append(file)
                            elif file[3:5] =='03':
                                labellist[4][2].append(file)
                            elif file[3:5] =='04':
                                labellist[4][3].append(file)
                            elif file[3:5] =='05':
                                labellist[4][4].append(file)
                            elif file[3:5] =='06':
                                labellist[4][5].append(file)
                        elif file[:2] == '06':
                            if file[3:5] == '01':
                                labellist[5][0].append(file)
                            elif file[3:5] == '02':
                                labellist[5][1].append(file)
                            elif file[3:5] == '03':
                                labellist[5][2].append(file)
                            elif file[3:5] == '04':
                                labellist[5][3].append(file)
                            elif file[3:5] == '05':
                                labellist[5][4].append(file)
                            elif file[3:5] == '06':
                                labellist[5][5].append(file)

            elif 'Image' in root:
                if os.path.isdir(root):
                    list = os.listdir(root)
                    for file in list:
                        if file[:2] == '01':  # Clothe ID가 01일때(ex : 01_02_F328_12)
                            if file[3:5] == '01':
                                imagelist[0][0].append(file)
                            elif file[3:5] == '02':
                                imagelist[0][1].append(file)
                            elif file[3:5] == '03':
                                imagelist[0][2].append(file)
                            elif file[3:5] == '04':
                                imagelist[0][3].append(file)
                            elif file[3:5] == '05':
                                imagelist[0][4].append(file)
                            elif file[3:5] == '06':
                                imagelist[0][5].append(file)
                        elif file[:2] == '02':
                            if file[3:5] == '01':
                                imagelist[1][0].append(file)
                            elif file[3:5] == '02':
                                imagelist[1][1].append(file)
                            elif file[3:5] == '03':
                                imagelist[1][2].append(file)
                            elif file[3:5] == '04':
                                imagelist[1][3].append(file)
                            elif file[3:5] == '05':
                                imagelist[1][4].append(file)
                            elif file[3:5] == '06':
                                imagelist[1][5].append(file)
                        elif file[:2] == '03':
                            if file[3:5] == '01':
                                imagelist[2][0].append(file)
                            elif file[3:5] == '02':
                                imagelist[2][1].append(file)
                            elif file[3:5] == '03':
                                imagelist[2][2].append(file)
                            elif file[3:5] == '04':
                                imagelist[2][3].append(file)
                            elif file[3:5] == '05':
                                imagelist[2][4].append(file)
                            elif file[3:5] == '06':
                                imagelist[2][5].append(file)
                        elif file[:2] == '04':
                            if file[3:5] == '01':
                                imagelist[3][0].append(file)
                            elif file[3:5] == '02':
                                imagelist[3][1].append(file)
                            elif file[3:5] == '03':
                                imagelist[3][2].append(file)
                            elif file[3:5] == '04':
                                imagelist[3][3].append(file)
                            elif file[3:5] == '05':
                                imagelist[3][4].append(file)
                            elif file[3:5] == '06':
                                imagelist[3][5].append(file)
                        elif file[:2] == '05':
                            if file[3:5] == '01':
                                imagelist[4][0].append(file)
                            elif file[3:5] == '02':
                                imagelist[4][1].append(file)
                            elif file[3:5] == '03':
                                imagelist[4][2].append(file)
                            elif file[3:5] == '04':
                                imagelist[4][3].append(file)
                            elif file[3:5] == '05':
                                imagelist[4][4].append(file)
                            elif file[3:5] == '06':
                                imagelist[4][5].append(file)
                        elif file[:2] == '06':
                            if file[3:5] == '01':
                                imagelist[5][0].append(file)
                            elif file[3:5] == '02':
                                imagelist[5][1].append(file)
                            elif file[3:5] == '03':
                                imagelist[5][2].append(file)
                            elif file[3:5] == '04':
                                imagelist[5][3].append(file)
                            elif file[3:5] == '05':
                                imagelist[5][4].append(file)
                            elif file[3:5] == '06':
                                imagelist[5][5].append(file)

    #작은 갯수 위주로 전처리 진행하기.
    addResultList = []
    bCnt = 0

    # 리스트 합집함 가져와서
    # 저장하는 로직으로 변경.
    # 찾아오면서 저장 X
    imgLen=0
    lblLen=0
    # 3시 22분 시작
    c=0
    for i in range(6):
        for j in range(6):
            imgLen+=len(imagelist[i][j])
            lblLen+=len(labellist[i][j])


    sCnt=min(imgLen, lblLen)
    print(f"총합 갯수 :{(imgLen , lblLen)}")
    bCnt=0

    #imagelist.sort(key=natural_keys())

    for i in range(6):
        for j in range(6):
            for img in imagelist[i][j]:
                print('\r', f'{bCnt}/{sCnt}, {round(bCnt/sCnt*100, 2)}%', end='')
                find_dir = img.split('/')[-1].split('.')[0]
                for label in labellist[i][j]:

                    if find_dir == label.split('/')[-1].split('.')[0]:
                        #조건문 test가 아니면 삭제 .
                        if '500' in label.split('/')[-1].split('.')[0]:
                            result = os.path.join(passDir1, label.split('.')[0][-7:-3], 'json', label) + ":" + os.path.join(
                                passDir2, label.split('.')[0][-7:-3], 'Image', img)

                            thread=Thread(target=txtWrite,args=(result,))
                            thread.start()
                            break

                bCnt += 1

    print("분류 및 매칭 완료")

    del addResultList
    print("success")
    return 0
def txtWrite(text):
    semaphore.acquire()
    print(f'{text}\n')

    # with open('./input_data/almost_full_label_data2.txt', 'a') as f:
    #     f.write(f"{text}\n")

    # f.close()
    semaphore.release()
    return 0
# def txtWrite(text):
#
#     with open('./input_data/almost_full_label_data.txt', 'a') as f:
#         for i in text:
#             f.write(f"{i}\n")
#     f.close()
#     return 0


#테스트 베드
res = txtWrite(imageDataLoad(readPath='/home/inviz/Data/Dataset/'))


if res == 0:

    print("Done")


