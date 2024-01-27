# -*- coding: utf-8 -*-
# inference.py

# # 만약 이미지가 너무 커서 메모리가 부족한 오류가 난다면 밑의 주석을 제거해주세요

'''
from PIL import Image
import os

# 이미지 파일이 있는 폴더 경로
folder_path = '../input/house'

# 크기를 조정할 폴더
resized_folder_path = "../input/house"

# 이미지 크기를 조정할 비율 (예: 0.5는 50%로 크기를 줄임)
resize_ratio = 0.5

# 폴더 내의 모든 이미지 파일에 대해 크기 조정
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        # 이미지 파일 로드
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)

        # 이미지 크기 조정
        width, height = img.size
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)
        resized_img = img.resize((new_width, new_height))

        # 크기를 조정한 이미지 저장
        new_filename = filename
        new_image_path = os.path.join(resized_folder_path, new_filename)
        resized_img.save(new_image_path)

print("이미지 크기 조정이 완료되었습니다.")

# 이미지 파일이 있는 폴더 경로
folder_path = '../input/tree'

# 크기를 조정할 폴더
resized_folder_path = "../input/tree"

# 이미지 크기를 조정할 비율 (예: 0.5는 50%로 크기를 줄임)
resize_ratio = 0.5

# 폴더 내의 모든 이미지 파일에 대해 크기 조정
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        # 이미지 파일 로드
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)

        # 이미지 크기 조정
        width, height = img.size
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)
        resized_img = img.resize((new_width, new_height))

        # 크기를 조정한 이미지 저장
        new_filename = filename
        new_image_path = os.path.join(resized_folder_path, new_filename)
        resized_img.save(new_image_path)

print("이미지 크기 조정이 완료되었습니다.")

# 이미지 파일이 있는 폴더 경로
folder_path = '../input/person'

# 크기를 조정할 폴더
resized_folder_path = "../input/person"

# 이미지 크기를 조정할 비율 (예: 0.5는 50%로 크기를 줄임)
resize_ratio = 0.5

# 폴더 내의 모든 이미지 파일에 대해 크기 조정
for filename in os.listdir(folder_path):
    if filename.endswith((".png", ".jpg", ".jpeg")):
        # 이미지 파일 로드
        image_path = os.path.join(folder_path, filename)
        img = Image.open(image_path)

        # 이미지 크기 조정
        width, height = img.size
        new_width = int(width * resize_ratio)
        new_height = int(height * resize_ratio)
        resized_img = img.resize((new_width, new_height))

        # 크기를 조정한 이미지 저장
        new_filename = filename
        new_image_path = os.path.join(resized_folder_path, new_filename)
        resized_img.save(new_image_path)

print("이미지 크기 조정이 완료되었습니다.")
'''

"""# house"""

import pandas as pd
from ultralytics import YOLO

model1 = YOLO(model='best_house.pt', task='detect')

image_path = '../input/house'
results = model1.predict(source=image_path,
                         conf=0.4,
                         iou=0.7
                         )

pred_house_df = pd.read_csv('../output/test_house.csv')

for r in results:
    id = r.path.split('/')[-1].replace('.jpg', '') # 파일명
    cls = r.boxes.cls.int()
    # 이미지당 결과에서 좌표 정보 추출
    # xywh : 바운딩 박스의 중심 x 좌표, 바운딩 박스의 중심 y 좌표, 바운딩 박스의 너비 (width), 바운딩 박스의 높이 (height)
    boxes = r.boxes.xywhn  # normalization
    count = 1
    df = {'id':id, 'door_yn':'n', 'loc':'center', 'roof_yn':'n', 'window_cnt':'absence', 'size':'middle'}

    # 클래스별로 좌표 정보 추출
    for c in cls.unique():
        class_boxes = boxes[cls==c]
        if c == 0:  # house
            count = len(class_boxes)
            # loc
            if class_boxes[0][0] < 1/3:
                df['loc'] = 'left'
            elif class_boxes[0][0] > 2/3:
                df['loc'] = 'right'
            # size
            if class_boxes[0][2] * class_boxes[0][3] >= 0.6: # 0.6
                df['size'] = 'big'
            elif class_boxes[0][2] * class_boxes[0][3] < 0.16: # 0.16
                df['size'] = 'small'
        if c == 1:  # roof
            df['roof_yn'] = 'y'
        if c == 2:  # window
            if 0 < len(class_boxes)/count <= 2:
                df['window_cnt'] = '1 or 2'
            elif len(class_boxes)/count > 2:
                df['window_cnt'] = 'more than 3'
        if c == 3:  # door
            df['door_yn'] = 'y'
    pred_house_df.loc[pred_house_df['id'] == id] = [df['id'], df['door_yn'], df['loc'], df['roof_yn'], df['window_cnt'], df['size']]

pred_house_df.to_csv('../output/test_house.csv', index=False, mode='w')

"""# tree"""

model2 = YOLO(model='best_tree.pt', task='detect')

image_path = '../input/tree'
results = model2.predict(source=image_path,
                         conf=0.4,
                         iou=0.7,
                         )

pred_tree_df = pd.read_csv('../output/test_tree.csv')

for r in results:
    id = r.path.split('/')[-1].replace('.jpg', '') # 파일명
    cls = r.boxes.cls.int()
    # 이미지당 결과에서 좌표 정보 추출
    # xywh : 바운딩 박스의 중심 x 좌표, 바운딩 박스의 중심 y 좌표, 바운딩 박스의 너비 (width), 바운딩 박스의 높이 (height)
    boxes = r.boxes.xywhn  # normalization

    df = {'id':id, 'branch_yn':'n', 'root_yn':'n', 'crown_yn':'n', 'fruit_yn':'n', 'gnarl_yn':'n', 'loc':'center', 'size':'middle'}

    for c in cls.unique():
        class_boxes = boxes[cls==c]
        if c == 0:  # tree
            # loc
            if class_boxes[0][0] < 1/3:
                df['loc'] = 'left'
            elif class_boxes[0][0] > 2/3:
                df['loc'] = 'right'
            # size
            if class_boxes[0][2] * class_boxes[0][3] >= 0.6:
                df['size'] = 'big'
            elif class_boxes[0][2] * class_boxes[0][3] < 0.16:
                df['size'] = 'small'
        if c == 1:  # ganrl
            df['gnarl_yn'] = 'y'
        if c == 2:  # crown
            df['crown_yn'] = 'y'
        if c == 3:  # branch
            df['branch_yn'] = 'y'
        if c == 4:  # root
            df['root_yn'] = 'y'
        if c == 5:  # fruit
            df['fruit_yn'] = 'y'

    pred_tree_df.loc[pred_tree_df['id'] == id] = [df['id'], df['branch_yn'], df['root_yn'], df['crown_yn'], df['fruit_yn'], df['gnarl_yn'], df['loc'], df['size']]

pred_tree_df.to_csv('../output/test_tree.csv', index=False, mode='w')

"""# person"""

model3 = YOLO(model='best_person.pt', task='detect')

image_path = '../input/person'
results = model3.predict(source=image_path,
                         conf=0.4,
                         iou=0.7,
                         )

pred_person_df = pd.read_csv('../output/test_person.csv')

for r in results:
    id = r.path.split('/')[-1].replace('.jpg', '') # 파일명
    cls = r.boxes.cls.int()
    # 이미지당 결과에서 좌표 정보 추출
    # xywh : 바운딩 박스의 중심 x 좌표, 바운딩 박스의 중심 y 좌표, 바운딩 박스의 너비 (width), 바운딩 박스의 높이 (height)
    boxes = r.boxes.xywhn  # normalization

    df = {'id':id, 'eye_yn':'n', 'leg_yn':'n', 'loc':'center', 'mouth_yn':'n', 'size':'middle', 'arm_yn':'n'}

    for c in cls.unique():
        class_boxes = boxes[cls==c]
        if c == 0:  # person
            # loc
            if class_boxes[0][0] < 1/3:
                df['loc'] = 'left'
            elif class_boxes[0][0] > 2/3:
                df['loc'] = 'right'
            # size
            if class_boxes[0][2] * class_boxes[0][3] >= 0.4: # 0.4
                df['size'] = 'big'
            elif class_boxes[0][2] * class_boxes[0][3] < 0.16: # 0.16
                df['size'] = 'small'
        if c == 1:  # eye
            df['eye_yn'] = 'y'
        if c == 2:  # mouth
            df['mouth_yn'] = 'y'
        if c == 3:  # arm
            df['arm_yn'] = 'y'
        if c == 4:  # leg
            df['leg_yn'] = 'y'

    pred_person_df.loc[pred_person_df['id'] == id] = [df['id'], df['eye_yn'], df['leg_yn'], df['loc'], df['mouth_yn'], df['size'], df['arm_yn']]

pred_person_df.to_csv('../output/test_person.csv', index=False, mode='w')

