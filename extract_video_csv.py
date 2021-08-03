import os
import cv2
import numpy as np
import pandas as pd
import argparse
import time
from FeatureExtractor import PoseEstimator, FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--video_dir', help='Directory contains all the videos', required=True)
parser.add_argument('--csv_dir', help='Directory for saving the output csv', required=True)
parser.add_argument('--output_dir', help='Directory for saving the output video', required=True)

args = parser.parse_args()
VIDEO_DIR = args.video_dir
CSV_DIR = args.csv_dir
OUTPUT_DIR = args.output_dir

pose_estimator = PoseEstimator()
feature_extractor = FeatureExtractor()
columns = ['HW_ratio', 'Spine_ratio', 'Body_tilt_angle', 'Neck_F_distance', 'Hip_F_distance',
           'Head_Acc', 'Neck_Acc', 'Spine_Acc', 
           'Spine_def', 'Waist_def', 'R_Thigh_def', 'L_Thigh_def', 'R_Calf_def', 'L_Calf_def']

def detect_img(image):
    height, width, _ = image.shape
    keypoints, output_img = pose_estimator.pose_estimation(image)
    
    if type(keypoints) != np.ndarray: #如果沒偵測到人的話 #如果沒偵測到人的話
        return [list(np.zeros((1,15))[0])], image 
    
    features = []
    for i in range(0, len(keypoints)):
        bbox, hw_ratio = feature_extractor.cal_HW_ratio(keypoints[i])
        spine_ratio = feature_extractor.cal_spine_ratio(keypoints[i])
        body_tilt = feature_extractor.cal_body_tilt_angle(keypoints[i])
        neck_hip_feet = feature_extractor.cal_neck_hip_feet_distance(keypoints[i], height, width)
        deflection_angles = feature_extractor.cal_deflection_angle(keypoints[i])
        accerlations = feature_extractor.cal_acceleration(keypoints[i], height, width)
        
        # 1代表Human is detected
        features.append([1,hw_ratio, spine_ratio, body_tilt] + neck_hip_feet + accerlations + deflection_angles)
        cv2.rectangle(output_img, bbox[0], bbox[1], (0,255,0), 2) #Draw BBox on Image
    
    return features, output_img

def output_csv_video_from_video(video_path, csv_folder, output_folder):
    start = time.time()
    
    ## Get Info from video path
    file = video_path.split('/')[-1].split('.')[0]
    try:
        folder = file.split('-')[1]
    except:
        folder = file
    frame_num = 1
    features = []
    output_path = os.path.join(output_folder, file+'.avi')
    csv_path = os.path.join(csv_folder, file+'.csv')
    
    ## Open Video File
    video = cv2.VideoCapture(video_path)
    imW = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    imH = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ## Output Video
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (imW, imH))
    
    ## Run main process
    while(video.isOpened()):
        ret, frame = video.read()
        if ret == False:
            break
        #print(file, folder, frame_num)
        feature, output_img = detect_img(frame)
        features.append([file, folder, frame_num]+feature[0])# 目前每個frame只取一個人
        cv2.putText(output_img, str(frame_num), (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4) # Put frame num on img
        out.write(output_img) # Write output video
        frame_num += 1
        #plt.imshow(output_img)

    ## Clean up
    video.release()
    out.release()

    ## Output csv file
    feature_columns = ['File','Folder','Frame_num','Human_Detected','HW_ratio', 'Spine_ratio', 'Body_tilt_angle',
                           'Neck_F_distance', 'Hip_F_distance','Head Acc','Neck Acc', 'Spine Acc',
                           'Spine_def','Waist_def','RThigh_def' ,'LThigh_def','RCalf_def', 'LCalf_def']
    df = pd.DataFrame(features, columns=feature_columns)
    df.to_csv(csv_path)
    end = time.time()
    print('Run Time: {}s'.format(end-start))
    

if __name__ == '__main__':
    video_paths = os.listdir(VIDEO_DIR)
    try:
        video_paths.remove('.DS_Store')
    except:
        pass
    video_paths.sort()
    video_paths = [os.path.join(VIDEO_DIR,p) for p in video_paths]
    
    for p in video_paths:
        print(p)
        output_csv_video_from_video(p, CSV_DIR, OUTPUT_DIR)