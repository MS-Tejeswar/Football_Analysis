import sys
sys.path.append('../')
from utils import get_center,get_width, measure

class PlayerBallAssigner():
    def __init__(self):
        self.max_player_ball_distance=70
        
    def assign_ball_to_player(self, players, ball_bbox):
        ball_pos= get_center(ball_bbox)
        mini=9999
        ap=-1
        for pid, player in players.items():
            player_bbox= player['bbox']
            d_left=measure((player_bbox[0],player_bbox[-1]),ball_pos)
            d_right=measure((player_bbox[2],player_bbox[-1]),ball_pos)
            d=min(d_left,d_right)
            
            if(d<self.max_player_ball_distance and d<mini):
                mini=d
                ap=pid
        return ap