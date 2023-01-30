# 영상을 사진으로 변환
# 초단위로 변환

import cv2
import os
import math


vidcap = cv2.VideoCapture('C:/Users/user/Downloads/123.mp4')

count = 0

while (vidcap.isOpened()):
    # read()는 grab()와 retrieve() 두 함수를 한 함수로 불러옴
    # 두 함수를 동시에 불러오는 이유는 프레임이 존재하지 않을 때
    # grab() 함수를 이용하여 return false 혹은 NULL 값을 넘겨 주기 때문
    ret, image = vidcap.read()
    
    
    # 20이면 2초에 1개씩 저장 /  10이면 1초에 1개씩 저장

    if (int(vidcap.get(1)) % 20 == 0):
        print('Saved frame number : ' + str(int(vidcap.get(1))))
        cv2.imwrite("C:/Users/user/Desktop/김서아/22_hf246/차선인식 - 서아/test/%d.jpg" % count, image)
        print('Saved frame%d.jpg' % count)
        count += 1

vidcap.release()