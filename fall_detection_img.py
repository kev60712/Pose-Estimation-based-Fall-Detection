import os
import cv2
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from FeatureExtractor import PoseEstimator, FeatureExtractor


'''==== Parser ===='''
parser = argparse.ArgumentParser()
parser.add_argument('--input_img', help='Input Image Path', required=True)
parser.add_argument('--output_img', help='Output Image Path', required=True)
args = parser.parse_args()
INPUT_IMG = args.input_img
OUTPUT_IMG = args.output_img

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

'''===== Function ===='''
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

if __name__ == '__main__':    
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
    img = cv2.imread(INPUT_IMG)
    output = fall_detect_img(img)
    cv2.imwrite(OUTPUT_IMG, output)
    print('Done')
