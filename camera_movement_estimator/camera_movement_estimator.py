import pickle
import os
import cv2
import numpy as np
import sys
sys.path.append('../')
from utils import measure, measure_xy_dist
class CameraMovementEstimator():
    def __init__(self, frame):
        self.minimum_distance=5
        
        first_frame_gs=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features=np.zeros_like(first_frame_gs)
        mask_features[:,0:20]=1
        mask_features[:,900:1050]=1
        
        self.lk_params=dict(
            winSize=(15,15),
            maxLevel=2,
            criteria =(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
            
        )
        
        self.features=dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize=7,
            mask=mask_features
        )
    
    def add_adjust_pos_tracks(self, tracks, cme_per_frame):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for tid, tinfo in track.items():
                    position = tinfo['position']
                    camera_movement = cme_per_frame[frame_num]
                    pos_adjusted= (position[0]- camera_movement[0], position[1] - camera_movement[1])
                    tracks[object][frame_num][tid]['position_adjusted']=pos_adjusted
    
    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)
            
        camera_movement=[[0,0]]*len(frames)
        old_gray=cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)
        
        for frame_num in range(1, len(frames)):
            frame=frames[frame_num]
            frame_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            new_features, status, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, old_features, None, **self.lk_params)
            max_dist=0
            camera_movement_x, camera_movement_y=0,0
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point=new.ravel()
                old_features_point=old.ravel()
                
                distance=measure(new_features_point, old_features_point) 
                if distance>max_dist:
                    max_dist=distance
                    camera_movement_x, camera_movement_y=measure_xy_dist(old_features_point, new_features_point)  
            if max_dist>self.minimum_distance:
                camera_movement[frame_num]=[camera_movement_x, camera_movement_y]
                old_features=cv2.goodFeaturesToTrack(frame_gray, **self.features)
                
            old_gray=frame_gray.copy()
        if stub_path is not None:
            pickle.dump(camera_movement, open(stub_path,'wb'))
        return camera_movement
    
    def draw_camera_movement(self, frames, cm_per_frame):
        output_frames=[]
        for frame_num, frame in enumerate(frames):
            frame=frame.copy()
            overlay=frame.copy()
            cv2.rectangle(overlay, (0,0), (500,100), (255,255,255), -1)
            alpha=0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
            x_movement, y_movement = cm_per_frame[frame_num]
            frame=cv2.putText(frame, f"X: {x_movement:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            frame=cv2.putText(frame, f"Y: {y_movement:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

            output_frames.append(frame)
        return output_frames