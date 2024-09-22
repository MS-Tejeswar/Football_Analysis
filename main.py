from utils import read_vid,save_vid
from trackers import Tracker
import cv2
from team_assignment import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import Estimator
frames=read_vid('input_vid/08fd33_4.mp4')


tracker= Tracker('models/best.pt')
tracks=tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/tracker_pickle.pkl')


tracker.add_pos_to_track(tracks)
#camera movement estimator
cme=CameraMovementEstimator(frames[0])
cm_per_frame=cme.get_camera_movement(frames, read_from_stub=True, stub_path='stubs/camera_movement_pickle.pkl')
cme.add_adjust_pos_tracks(tracks, cm_per_frame)

#View Transformer
view_trans=ViewTransformer()
view_trans.add_transformed_position_tracks(tracks)

#interpolate ball positions
tracks['ball']=tracker.interpolate_ball_positions(tracks['ball'])


#speed and dist estimator
estimator=Estimator()
estimator.add_speed_and_distance(tracks)
#assign player teams
team_assigner=TeamAssigner()
team_assigner.assign_team_color(frames[0], tracks['players'][0])


for frame_num, player_track in enumerate(tracks['players']):
    for pid, track in player_track.items():
        team=team_assigner.get_player_team(frames[frame_num], track['bbox'], pid)
        tracks['players'][frame_num][pid]['team']=team
        tracks['players'][frame_num][pid]['team_color']=team_assigner.team_colors[team]
        
#assign ball acquisition
pa=PlayerBallAssigner()
team_ball_control=[]
for frame_num, player_track in enumerate(tracks['players']):
    ball_bbox=tracks['ball'][frame_num][1]['bbox']
    ap=pa.assign_ball_to_player(player_track, ball_bbox)
    
    if(ap!=-1):
        tracks['players'][frame_num][ap]['has_ball']=True
        team_ball_control.append(tracks['players'][frame_num][ap]['team'])
    else:
        team_ball_control.append(team_ball_control[-1])
team_ball_control = np.array(team_ball_control)   


op=tracker.draw_annotations(frames, tracks, team_ball_control)
op = cme.draw_camera_movement(op, cm_per_frame)

#draw speed and distance
estimator.draw_sand(op, tracks)
save_vid(op,'output_vids/track13.avi')