import sys
sys.path.append('../')
from utils import measure_xy_dist, get_foot_pos, measure
import cv2
class Estimator:

    def __init__(self):
        self.frame_window=5
        self.frame_rate=24
        
    def add_speed_and_distance(self, tracks):
        total_dist_covered={}
        for object, object_tracks in tracks.items():
            if(object=='ball' or object=='referee'):
                continue
            no_of_frames=len(object_tracks)
            for frame_num in range(0,no_of_frames,self.frame_window):
                last_frame = min(frame_num+self.frame_window, no_of_frames-1)
                for tid, tinfo in object_tracks[frame_num].items():
                    if tid not in object_tracks[last_frame]:
                        continue
                    start=object_tracks[frame_num][tid]['position_transformed']
                    end=object_tracks[last_frame][tid]['position_transformed']
                    
                    if start is None or end is None:
                        continue
                    distance = measure(start, end)
                    time_elapsed= (last_frame-frame_num)/self.frame_rate
                    speed = distance/time_elapsed
                    skmph=speed*3.6
                    if( object not in total_dist_covered):
                        total_dist_covered[object]={} 
                    if tid not in total_dist_covered[object]: 
                        total_dist_covered[object][tid]=0
                        
                    total_dist_covered[object][tid]+=distance
                    
                    for frame_num_batch in range(frame_num,last_frame):
                        if tid not in tracks[object][frame_num_batch]:
                            continue    
                        tracks[object][frame_num_batch][tid]['speed']=skmph
                        tracks[object][frame_num_batch][tid]['distance']=total_dist_covered[object][tid]
    def draw_sand(self, frames, tracks):
        output_frames=[]
        for frame_num, frame in enumerate(frames):   
            for object, object_tracks in tracks.items():
                if object=='ball' or object=='referees' or object=='referee':
                    continue     
                for _, tinfo in object_tracks[frame_num].items():
                    print(tinfo)
                    if "speed" in tinfo:
                        speed=tinfo.get('speed', None)   
                        dist=tinfo.get('distance', None)
                        if speed is None or dist is None: 
                            continue   
                        bbox=tinfo.get('bbox', None)    
                        position=get_foot_pos(bbox)
                        position=list(position)
                        position[1]+=40
                        
                        position= tuple(map(int,position))
                        cv2.putText(frame, f"{speed:.2f} km/h", position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
                        cv2.putText(frame, f"{dist:.2f} m", (position[0],position[1]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
            output_frames.append(frame)
        return output_frames