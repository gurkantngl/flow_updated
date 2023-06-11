import cv2
import os
import numpy as np
#define fps variable

FPS = 1
MOVE_THRESH = 30

def get_tracking_points(old_gray, frame):
    mask = np.zeros_like(frame)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    thresh = len(p0[:int(len(p0) * 0.05)])
    if thresh <= 0:
        thresh = 1
    return mask, p0, thresh

def get_movement_flag(points):
    distances=[]
    for i in range(len(points[0])):
        d=[]
        first_point=points[0][i][0][1]
        for j in range(len(points)-1):
            # if j+1 not out of range
            try:                              
                second_point=points[j+1][i][0][1]
                dist=np.linalg.norm(first_point-second_point)
                d.append(dist)

            except:
                pass
        distances.append(d)

    increasing_count = 0
    distance_diff=[]
    for i, array in enumerate(distances):
        if len(array) > 0:
            if np.all(np.diff(array) > 0):
                    distance_diff.append(array[-1]-array[0])
                    increasing_count += 1

    #print(sum(distance_diff), move_threshold)
    if sum(distance_diff) > move_threshold:
        move_flag=True
        #print(sum(distance_diff), move_threshold)
    else:
        move_flag=False

    return move_flag


feature_params = dict(maxCorners=100, qualityLevel=0.1, minDistance=30, blockSize=27)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
color = np.random.randint(0, 255, (100, 3))

magnitude_thresh=20#20
move_flag=False
points=[]

folder_path = 'video/'

if __name__ == '__main__':

    files = os.listdir(folder_path)
    for file_name in files:
        if file_name.endswith('.mp4'):
            video_path = os.path.join(folder_path, file_name)

            cap = cv2.VideoCapture(video_path)
            ret, old_frame = cap.read()

            height, width = old_frame.shape[:2]
            center_x, center_y = width // 2, height // 2
            radius = width // 4

            circular_mask = np.zeros(old_frame.shape[:2], dtype=np.uint8)
            cv2.circle(circular_mask, (center_x, center_y), radius, (255), thickness=-1)

            masked_old_frame = cv2.bitwise_and(old_frame, old_frame, mask=circular_mask)
            old_gray = cv2.cvtColor(masked_old_frame, cv2.COLOR_BGR2GRAY)

            diag = np.sqrt((width ** 2) + (height ** 2))
            move_threshold = diag * 0.002#0.005


            mask, p0, thresh = get_tracking_points(old_gray, old_frame)
            move_count=0
            while True:
                
                ret, frame = cap.read()
                if not ret:
                    break

                if cap.get(cv2.CAP_PROP_POS_FRAMES) %FPS==0:

            
                    masked_frame = cv2.bitwise_and(frame, frame, mask=circular_mask)
                    frame_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)

                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    if p1 is not None and st is not None:

                        good_new = p1[st == 1]
                        good_old = p0[st == 1]

                        new_points = [(point,label[0]) for point,label in zip(good_new.reshape(-1, 2),st)]
                        old_points = [(point,label[0]) for point,label in zip(good_old.reshape(-1, 2),st)]
                
                        current_flow_vectors = good_new - good_old
                        current_flow_mag = np.linalg.norm(current_flow_vectors, axis=1)     
                        

                        if any(current_flow_mag > magnitude_thresh):
                            mask, p0, thresh = get_tracking_points(old_gray, frame)
                            points=[]
                            move_flag=False
                        else:
                            
                            p=[]
                            for point in range(len(new_points)):
                                p.append(new_points[point])
                            points.append(p)

                            if len(points) > 5:
                                points.pop(0)
                            
                                move_flag=get_movement_flag(points)
                                
                                for i, (new, old) in enumerate(zip(good_new, good_old)):
                                    a, b = new.ravel()
                                    c, d = old.ravel()
                                    mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
                                    frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)




                        if move_flag:
                            move_count+=1
                            cv2.putText(frame, 'HAREKET', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 2, cv2.LINE_AA)
                        else:
                            move_count=0
                                
                        
                        if move_count > MOVE_THRESH:
                            cv2.putText(frame, 'ALARM', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3, cv2.LINE_AA)
                            print(video_path, " ALARM VERİLDİ!")
                        img = cv2.add(frame, mask)
                        cv2.namedWindow('Combined', cv2.WINDOW_NORMAL)
                        cv2.resizeWindow('Combined', 1920, 1080)
                        cv2.imshow('Combined', img)

                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                        old_gray = frame_gray.copy()
                        p0 = good_new.reshape(-1, 1, 2)
                    else:
                        mask, p0, thresh = get_tracking_points(old_gray, frame)
                        points=[]
                        move_flag=False


                if cap.get(cv2.CAP_PROP_POS_FRAMES) % 1000 == 0:
                    mask, p0, thresh = get_tracking_points(old_gray, frame)
                    points=[]

                    move_flag=False