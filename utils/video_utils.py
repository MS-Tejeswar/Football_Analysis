import cv2

def read_vid(path):
    cap=cv2.VideoCapture(path)
    frames=[]
    while True:
        ret,frame=cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_vid(op_frames,op_path):
    fourcc= cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter(op_path, fourcc, 24, (op_frames[0].shape[1],op_frames[0].shape[0]))
    for frame in op_frames:
        out.write(frame)
        
    out.release()
