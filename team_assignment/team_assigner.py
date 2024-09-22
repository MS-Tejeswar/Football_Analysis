from sklearn.cluster import KMeans
class TeamAssigner:
    def __init__(self):
        self.team_colors={}
        self.player_team_dict={}
    
    def get_clustering_model(self, top_half):
        image_2d=top_half.reshape((-1,3))
        kmeans= KMeans(n_clusters=2, random_state=0).fit(image_2d)
        return kmeans
    
    def get_player_color(self, bbox, frame):
        image=frame[int(bbox[1]):int(bbox[3]),int(bbox[0]):int(bbox[2])]
        top_half=image[:int(image.shape[0]/2),:]
        
        kmeans=self.get_clustering_model(top_half)
        
        labels=kmeans.labels_
        
        clustered_image=labels.reshape(top_half.shape[0], top_half.shape[1])
        corner_cluster=[clustered_image[0][0],clustered_image[0][-1],clustered_image[-1][0],clustered_image[-1][-1]]
        np_cluster=max(set(corner_cluster), key=corner_cluster.count)
        p_cluster=1-np_cluster   
        p_color=kmeans.cluster_centers_[p_cluster] 
        return p_color
    def assign_team_color(self,frame, player_detections):
        player_colors=[]
        for _, player in player_detections.items():
            bbox=player['bbox']
            player_color=self.get_player_color(bbox, frame)
            player_colors.append(player_color)
        kmeans=KMeans(n_clusters=2, random_state=0).fit(player_colors)
        
        self.kmeans=kmeans
        
        self.team_colors[1]=kmeans.cluster_centers_[0]
        self.team_colors[2]=kmeans.cluster_centers_[1]
        
    def get_player_team(self, frame, player_bbox, pid):
        if pid in self.player_team_dict:
            return self.player_team_dict[pid]
        else:
            player_color=self.get_player_color(player_bbox, frame)
            team_id=self.kmeans.predict(player_color.reshape(1,-1))[0]+1
            
            self.player_team_dict[pid]=team_id
            
            return team_id
        
            