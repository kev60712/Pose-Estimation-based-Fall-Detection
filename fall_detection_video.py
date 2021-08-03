import os
import cv2
import timeit
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from FeatureExtractor import PoseEstimator, FeatureExtractor


'''==== Parser ===='''
parser = argparse.ArgumentParser()
parser.add_argument('--video', help='Video Path', required=True)
parser.add_argument('--output_dir', help='Output Folder', required=True)
args = parser.parse_args()
VIDEO_PATH = args.video
OUTPUT_DIR = args.output_dir

'''==== Create Model ===='''
# Get Train and Test Data
X = pd.read_csv('dataset/X_train.csv').drop(columns=['Unnamed: 0']).values
y = pd.read_csv('dataset/y_train.csv').drop(columns=['Unnamed: 0']).values
# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)
# KNN Model
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', p=1)
knn_model.fit(X, y)

'''==== Function ===='''
def fall_detect_img(image):
    height, width, _ = image.shape
    keypoints, output_img = pose_estimator.pose_estimation(image)
    bboxes = []
    features = []
    
    if type(keypoints) != np.ndarray: #如果沒偵測到人的話 #如果沒偵測到人的話
        return image
    
    for i in range(0, len(keypoints)):
        bbox, hw_ratio = feature_extractor.cal_HW_ratio(keypoints[i])
        spine_ratio = feature_extractor.cal_spine_ratio(keypoints[i])
        body_tilt = feature_extractor.cal_body_tilt_angle(keypoints[i])
        neck_hip_feet = feature_extractor.cal_neck_hip_feet_distance(keypoints[i], height, width)
        deflection_angles = feature_extractor.cal_deflection_angle(keypoints[i])
        accerlations = feature_extractor.cal_acceleration(keypoints[i], height, width)
        
        bboxes.append(bbox)
        features.append([hw_ratio, spine_ratio, body_tilt] + neck_hip_feet + accerlations + deflection_angles)

    for feature, b in zip(features, bboxes):
        xmin = int(b[0][0])
        ymin = int(b[0][1])
        ## 如果feature有任一值為NaN, return None
        if pd.isnull(feature).any():
            for i in [1,2,3,4,8,9,10,11,12,13]:
                if feature[i] == None:
                    feature[i] = 0
                    feature += [0]
                else:
                    feature += [1]
        else:
            feature += [1]*10
        
        # Fall Detection
        feature = sc.transform(np.array([feature]))
        label = knn_model.predict(feature)[0]

        # Draw Bbox
        cv2.rectangle(output_img, b[0], b[1], (0,255,0), 2)

        # Draw Label
        if label == 0:
            cv2.putText(output_img, 'Normal', (xmin-20, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        elif label == 1:
            cv2.putText(output_img, 'Fall', (xmin-20, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        elif label == 2:
            cv2.putText(output_img, 'Lying', (xmin-20, ymin-20), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
    
    return output_img

def fall_detect_video(video_path, output_folder):
    start = timeit.default_timer()
    # init
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
    file_name = video_path.split('/')[-1].split('.')[0]
    output_path = os.path.join(output_folder, file_name+'.avi')

    # Open Video file
    video = cv2.VideoCapture(video_path)
    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output Video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (imW, imH))
    
    # Run Deteciton on Video
    while(video.isOpened()):
        ret, frame = video.read()
        if not ret:
            print('Reach the end of the video')
            break
        
        output_img = fall_detect_img(frame)
        
        out.write(output_img)
    out.release()
    
    stop = timeit.default_timer()
    print('Time: ', stop - start)

if __name__ == '__main__':    
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
    fall_detect_video(VIDEO_PATH, OUTPUT_DIR)