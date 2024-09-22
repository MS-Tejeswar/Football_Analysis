from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
sys.path.append('../')
from utils import get_center,get_width, get_foot_pos
import cv2
import numpy as np
import pandas as pd
class Tracker:
    def __init__(self, model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack()   
    
    
    def add_pos_to_track(self, tracks):
        for object, object_tracks in tracks.items():    
            for frame_num, track in enumerate(object_tracks):
                for tid, tinfo in track.items():
                    bbox=tinfo['bbox']
                    if object=='ball':
                        position=get_center(bbox)
                    else:
                        position=get_foot_pos(bbox)
                    tracks[object][frame_num][tid]['position']=position
    def detect_frames(self, frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            detection=self.model.predict(frames[i:i+batch_size], conf=0.1, stream=True)
            detections.extend(detection)
        return detections
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            tracks=pickle.load(open(stub_path,'rb'))
            return tracks
        
        detections= self.detect_frames(frames)
        
        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }
        
        for frame_num, detection in enumerate(detections):
            class_names=detection.names
            class_inv={v:k for k,v in class_names.items()}
            detection_sv=sv.Detections.from_ultralytics(detection)
            
            
            #convert gk to player
            for object_ind,class_id in enumerate(detection_sv.class_id):
                if class_names[class_id]=='goalkeeper':
                    detection_sv.class_id[object_ind]=class_inv['player']
                    
                    
            #Track objects
            detection_with_tracks=self.tracker.update_with_detections(detection_sv)
            
            tracks["ball"].append({})
            tracks["players"].append({})
            tracks["referees"].append({})   
            
            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                cid=frame_detection[3]
                tid=frame_detection[4]
                
                if cid==class_inv['player']:
                    tracks["players"][frame_num][tid]={"bbox":bbox}
                    
                if cid==class_inv['referee']:
                    tracks["referees"][frame_num][tid]={"bbox":bbox}
                    
            for frame_detection in detection_sv:
                bbox=frame_detection[0].tolist()
                cid=frame_detection[3]
                if cid==class_inv['ball']:
                    tracks["ball"][frame_num][1]={"bbox":bbox}
        #    print(detection_sv)
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(tracks,f)
        return tracks
    
    
    def draw_triangle(self, frame, bbox, color):
        y=int(bbox[1])
        x,_=get_center(bbox)
        triangle_pts=np.array([
            [x, y], [x+10, y-20], [x-10, y-20]
        ])
        cv2.drawContours(frame, [triangle_pts], 0, color, -1)
        cv2.drawContours(frame, [triangle_pts], 0, (0,0,0), 2)
        return frame
    
    def draw_ellipse(self, frame, bbox, color, tid=None):
        y2=int(bbox[3])
        xc, yc=get_center(bbox)
        w=get_width(bbox)
        cv2.ellipse(
            frame,
            center=(xc, y2),
            axes=(int(w), int(0.35*w)),
            angle=0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )
        
        rw=40
        rh=20
        x1r=xc-rw/2
        x2r=xc+rw/2
        y1r=y2-rh/2+15
        y2r=y2+rh/2+15
        
        if tid is not None:
            cv2.rectangle(
                frame,
                pt1=(int(x1r), int(y1r)),
                pt2=(int(x2r), int(y2r)),
                color=color
            )
            x1text=x1r+15
            if tid>99:
                x1text-=10
            cv2.putText(frame, str(tid), (int(x1text), int(y1r+15)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        return frame
    
    def interpolate_ball_positions(self, ball_positions):
        ball_positions=[x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_bp=pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
        
        #interpolate missing values
        df_bp.interpolate(inplace=True)
        df_bp.bfill(inplace=True)
        
        ball_positions=[{1: {'bbox':x}} for x in df_bp.to_numpy().tolist()]
        
        return ball_positions
    
    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        #draw semi transparent rectangle
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
        alpha= 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        team1_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team2_num_frames=team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        t1=team1_num_frames/(team1_num_frames+team2_num_frames)
        t2=team2_num_frames/(team1_num_frames+team2_num_frames)
        
        cv2.putText(frame, f"Team 1: {t1*100:.2f}%", (1500, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2: {t2*100:.2f}%", (1500, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        return frame
    def draw_annotations(self, video_frames, tracks, team_ball_control):
        
        op=[]
        for frame_num,frame in enumerate(video_frames):
            frame=frame.copy()
            player_dict=tracks["players"][frame_num]
            ball_dict=tracks["ball"][frame_num]
            referee_dict=tracks["referees"][frame_num]
            
            #Draw player
            for tid, player in player_dict.items():
                color=player.get("team_color", (0,0,255))
                frame=self.draw_ellipse(frame, player["bbox"],color,tid)
                
                if(player.get('has_ball',False)):
                    frame=self.draw_triangle(frame, player["bbox"],(0,255,0))
                
            #Draw referr
            for tid, ref in referee_dict.items():
                frame=self.draw_ellipse(frame, ref["bbox"],(255,0,255))
                
            #draw ball
            for tid,ball in ball_dict.items():
                frame=self.draw_triangle(frame, ball["bbox"], (255,0,0))
                
            #draw team ball control
            frame=self.draw_team_ball_control(frame, frame_num, team_ball_control)   
                
            op.append(frame)
        return op
    