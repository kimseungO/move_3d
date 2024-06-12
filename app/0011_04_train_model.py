import pandas as pd
import numpy as np
import cv2
import os

# Define column names for your data
column_names = ['input_image', 'bbox', 'heading', 'type','3d_center_x', '3d_center_y', 'real_width', 'real_length']

# Load data from a CSV file using pandas
csv_file_path = 'data/0001_csv/0001_01_train.csv'
data = pd.read_csv(csv_file_path, names=column_names, header=None)

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, concatenate
from keras.applications import ResNet50

# ######################################################
# 모델 생성
# ######################################################
# 모델 입력 설정

num_classes = 1  # 객체 class 번호 (0:car or 1:bus or 2:truck)
# input_image = Input(shape=(224, 224, 3), name='input_image')  # RGB 이미지
# input_depth_map = Input(shape=(224, 224, 1), name='input_depth_map')  # 생성된 depth map 이미지 ==> depth map은 처음에는 빼고 실험하고 나중에 실험결과를 보고 추가
input_heading = Input(shape=(1,), name='input_heading') # heading (0~360)
input_bbox = Input(shape=(4,), name='input_bbox')  # 바운딩 박스 좌표 (x_min, y_min, x_max, y_max)
input_type = Input(shape=(num_classes,), name='input_type')  # 객체 타입에 대한 원-핫 인코딩 벡터

# ResNet50이 base model
# base_model = ResNet50(weights='imagenet', include_top=False)
# base_model.trainable = False  # 사전 훈련된 가중치를 사용하고 추가 학습을 방지

# 이미지 특징 추출
# image_features = base_model(input_image)
# image_features = GlobalAveragePooling2D()(image_features)

# Depth map 특징 추출
# depth_features = base_model(input_depth_map)
# depth_features = GlobalAveragePooling2D()(depth_features)

# 특징 결합 (이미지 특징, 뎁스 맵 특징, 바운딩박스, 객체 타입)
combined_features = concatenate([input_heading, input_bbox, input_type]) # , depth_features, image_features

# Fully connected layers 추가. 레이어 수는 변경 가능
fc_layer = Dense(4096, activation='relu')(combined_features)


# Add additional hidden layers with more nodes

#fc_layer = Dense(2048, activation='relu')(fc_layer)
fc_layer = Dense(1024, activation='relu')(fc_layer)
fc_layer = Dense(512, activation='relu')(fc_layer)
fc_layer = Dense(512, activation='relu')(fc_layer)
fc_layer = Dense(256, activation='relu')(fc_layer)
fc_layer = Dense(256, activation='relu')(fc_layer)
fc_layer = Dense(128, activation='relu')(fc_layer)
fc_layer = Dense(128, activation='relu')(fc_layer)
fc_layer = Dense(64, activation='relu')(fc_layer)
fc_layer = Dense(64, activation='relu')(fc_layer)
fc_layer = Dense(32, activation='relu')(fc_layer)  # Add a layer with 64 nodes
fc_layer = Dense(32, activation='relu')(fc_layer)  # Add a layer with 32 nodes

# Output Layer
output_3d_center = Dense(2, activation='linear', name='output_3d_center')(fc_layer)  # 중심의 x, y
output_3d_dims = Dense(2, activation='linear', name='output_3d_dims')(fc_layer)  # 넓이, 길이 
# output_3d_head = Dense(2, activation='linear', name='output_3d_head')(fc_layer)  # 넓이, 길이, 높이

# 모델 생성
model = Model(inputs=[input_heading, input_bbox, input_type], # input_depth_map, input_image
            outputs=[output_3d_center, output_3d_dims]) # output_3d_head

# 모델 컴파일
model.compile(optimizer='adam',
            loss={'output_3d_center': 'mse', 'output_3d_dims': 'mse'}, # , 'output_3d_head': 'mse'
            metrics={'output_3d_center': 'mae', 'output_3d_dims': 'mae'}) # , 'output_3d_head': 'mae'

# 모델 요약 출력
model.summary()

#######################################################

#######################################################
# 학습
#######################################################
import numpy as np
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
npyfolder = 'data/npy/1125/'
# train_images = np.load(npyfolder + 'train_images.npy')  # shape (num_samples, 224, 224, 3)
# train_depth_maps = np.load('path_to_train_depth_maps.npy')  # shape (num_samples, 224, 224, 1)
train_bboxes = np.load(npyfolder + 'train_bboxes.npy')  # shape (num_samples, 4)
train_types = np.load(npyfolder + 'train_types.npy')  # shape (num_samples, num_classes)
train_3d_centers = np.load(npyfolder + 'train_3d_centers.npy')  # shape (num_samples, 2)
train_3d_dims = np.load(npyfolder + 'train_3d_dims.npy')  # shape (num_samples, 2)
train_3d_head = np.load(npyfolder + 'train_3d_head.npy')  # shape (num_samples, 1)


optimizer = Adam(learning_rate=1e-4)
model.compile(optimizer=optimizer,
            loss={'output_3d_center': 'mse', 'output_3d_dims': 'mse'}, # , 'output_3d_head': 'mse'
            metrics={'output_3d_center': 'mae', 'output_3d_dims': 'mae'}) # , 'output_3d_head': 'mae'
batch_size = 32
epochs = 500
# Early stopping 설정
early_stopping = EarlyStopping(monitor='mse', patience=10, restore_best_weights=True)
history = model.fit(
    [train_3d_head, train_bboxes, train_types], # , train_depth_maps
    [train_3d_centers, train_3d_dims], # , train_3d_head
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2,
    callbacks=[early_stopping]
)
loss_history = history.history

model.save("data/train_output/1227/model/0001_01_train.keras")
print("model saved")


import pickle

# Save the training history using pickle
with open('data/train_output/1227/history/0001_01_train.pkl', 'wb') as file:
    pickle.dump(history.history, file)
print("history saved")
# import matplotlib.pyplot as plt