import timeit
import sys
import cv2
import os
from sys import platform
import pyopenpose as op
from matplotlib import pyplot as plt
import numpy as np
import time
from pandas import DataFrame
import argparse
import pandas as pd

class PoseEstimator:
    def __init__(self):
        # Custom Params (refer to include/openpose/flags.hpp for more parameters)
        self.params = dict()
        self.params["model_folder"] = "/openpose/models/"
        
        # Starting OpenPose 
        self.opWrapper = op.WrapperPython()
        self.opWrapper.configure(self.params)
        self.opWrapper.start()
        # Process img
        self.datum = op.Datum()
        
    def pose_estimation(self, img):
        self.datum.cvInputData = img
        self.opWrapper.emplaceAndPop(op.VectorDatum([self.datum]))
        
        return self.datum.poseKeypoints, self.datum.cvOutputData
        #return self.datum.poseKeypoints
        

class FeatureExtractor:
    def __init__(self):
        self.frame_num = 0
        self.cal_per_frames = 5 #目前已每5frame計算一次，大約是0.25s，但其實不是，之後可能要改
        self.pre_head = self.pre_spine = self.pre_neck = self.head = self.spine = self.neck= None
        
    def add_frame_num(self):
        self.frame_num += 1
    
    def cal_HW_ratio(self, keypoints): 
        df = pd.DataFrame(keypoints)
        df = df[df[2]!=0].drop(columns=[2])
        xmax, ymax = df.max()
        xmin, ymin = df.min()

        width = xmax - xmin
        height = ymax - ymin

        return [(int(xmin), int(ymin)),(int(xmax), int(ymax))], height/width
    
    def cal_acceleration(self, keypoints, height, width):
        self.frame_num+=1
        self.head = keypoints[0][1]
        self.neck = keypoints[1][1]
        self.spine = keypoints[8][1]
        
        #print('Frame:{}, Pre_H:{}, H:{}, Pre_S:{}, S:{}'.format(self.frame_num, self.pre_head, self.head, self.pre_spine, self.spine))
        
        if self.pre_head==None and self.pre_spine==None: #第一次執行
            self.pre_head = self.head
            self.pre_neck = self.neck
            self.pre_spine = self.spine
            return [0,0,0]
        elif self.frame_num < self.cal_per_frames: #還沒超過5frame
            return [0,0,0]
        
        elif self.frame_num >= self.cal_per_frames:
            #計算加速度
            adjust_value = int(854*height/width)/height #調整數值，讓資料不會因為不同shape有很大差異
            #如果pre_head或head其中一個沒偵測到，加速度為0，以此類推
            head_acc = 0 if (self.head == 0 or self.pre_head == 0) else adjust_value*(self.head - self.pre_head)/5
            neck_acc = 0 if (self.neck == 0 or self.pre_neck == 0) else adjust_value*(self.neck - self.pre_neck)/5
            spine_acc = 0 if (self.spine == 0 or self.pre_spine == 0) else adjust_value*(self.spine - self.pre_spine)/5
            #print('Head_A: {}, Spine_A: {}'.format(head_acc, spine_acc))
            #print('\n')
            
        if self.frame_num%self.cal_per_frames == 0:
            self.pre_head = self.head
            self.pre_neck = self.neck
            self.pre_spine = self.spine
        
        return [head_acc, neck_acc, spine_acc]
    
    def cal_deflection_angle(self, keypoints):
        spine_vec = self.cal_keypoints_vec(keypoints, 1, 8)
        waist_width_vec = self.cal_keypoints_vec(keypoints, 9,12)
        r_thigh_vec = self.cal_keypoints_vec(keypoints, 9, 10)
        l_thigh_vec = self.cal_keypoints_vec(keypoints, 12, 13)
        r_calf_vec = self.cal_keypoints_vec(keypoints, 10, 11)
        l_calf_vec = self.cal_keypoints_vec(keypoints, 13, 14)
        g_vec = np.array([0,-100])
        angles = []
        
        for vec in [spine_vec, waist_width_vec, r_thigh_vec, l_thigh_vec, r_calf_vec, l_calf_vec]:
            if type(vec) != np.ndarray:
                angles.append(None)
            else:
                cos = vec.dot(g_vec)/(np.linalg.norm(vec)*np.linalg.norm(g_vec))
                angle = np.arccos(cos)*360/(2*np.pi)
                angles.append(angle)
                
        return angles
        
        
    def cal_keypoints_vec(self, keypoints, start, end):
        if keypoints[start][2] != 0 and keypoints[end][2]:
            vec = np.array([keypoints[start][0]-keypoints[end][0], keypoints[start][1]-keypoints[end][1]])
            return vec
        else:
            return None
    
    def cal_spine_ratio(self, keypoints):
        if keypoints[1][2]==0 or keypoints[8][2]==0 or keypoints[9][2]==0 or keypoints[12][2]==0:
            return None
        
        spine_vec = np.array([keypoints[1][0]-keypoints[8][0], keypoints[1][1]-keypoints[8][1]])
        waist_width_vec = np.array([keypoints[9][0]-keypoints[12][0], keypoints[9][1]-keypoints[12][1]])
    
        ratio = np.linalg.norm(spine_vec)/np.linalg.norm(waist_width_vec)
    
        return ratio
    
    
    def cal_body_tilt_angle(self, keypoints):
        if keypoints[1][2] == 0:
            return None
        
        neck_keypoint = [keypoints[1][0], keypoints[1][1]] #keypoint 1
        keypoint_list = []
        angle_list = []
        
        if keypoints[8][2] != 0: 
            keypoint_list.append([keypoints[8][0], keypoints[8][1]]) #Add C_Hip
        if keypoints[10][2]!=0 and keypoints[13][2]!=0:
            keypoint_list.append([(keypoints[10][0]+keypoints[13][0])/2, (keypoints[10][1]+keypoints[13][1])/2]) #Add C_Knee
        if keypoints[11][2]!=0 and keypoints[14][2]!=0:
            keypoint_list.append([(keypoints[11][0]+keypoints[14][0])/2, (keypoints[11][1]+keypoints[14][1])/2]) #Add C_Ankle
        
        if len(keypoint_list) != 0:
            for k in keypoint_list:
                tan = abs(neck_keypoint[1]-k[1])/abs(neck_keypoint[0]-k[0])
                angle_list.append(np.arctan(tan)*360/(2*np.pi))
            return min(angle_list)
        else:
            return None
    
    def cal_neck_hip_feet_distance(self, keypoints, height, width):
        ## 由於對所有圖片進行resize會造成圖片失真，因此設置調整值來調整不同size圖片的問題
        ## 以寬為854固定，調整高，在對距離做調整
    
        if keypoints[11][2] == 0 and keypoints[14][2] == 0:
            return [None, None]
        elif keypoints[11][2] == 0 or keypoints[14][2]==0:
            feet_y = keypoints[11][1] + keypoints[14][1]
        else:
            feet_y = (keypoints[11][1] + keypoints[14][1])/2
        
        adjust_value = int(854*height/width)/height #調整數值，讓資料不會因為不同shape有很大差異
            
        neck_distance = feet_y-keypoints[1][1] if keypoints[1][2] != 0 else None
        hip_distance  = feet_y-keypoints[8][1] if keypoints[8][2] != 0 else None
        
        neck_distance = neck_distance if neck_distance == None else neck_distance*adjust_value
        hip_distance = hip_distance if hip_distance == None else hip_distance*adjust_value
        
        #print(keypoints[1][1], keypoints[8][1], feet_y)
        return [neck_distance, hip_distance]

#if __name__ == '__main__': 
    #pose_estimator = PoseEstimator()
    #img = cv2.imread('image/test/image_0071.png')
    #keypoints, output_img = pose_estimator.pose_estimation(img)
    #print('偵測到人數：', len(keypoints))
    

