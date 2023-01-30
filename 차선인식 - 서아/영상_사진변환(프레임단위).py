# Mapping 도전 
# 구상 / 10X10 행렬 만들어 10프레임 동안의 움직임 분석해 위에 그림

import cv2
import os
import math

#######################동영상 -> 사진 파일 저장#########################

def frame(filename, portal): 
    vid=cv2.VideoCapture(filename) #비디오 보기
    count=0
    while True:
        yes, image=vid.read() #영상을 사진 단위로 받아오기
        if not yes: #영상이 끝나면
            break
        count+=1
        cv2.imwrite(portal+'/'+str(count)+'.png',image) #지정된 경로에 저장
        if count==200:
            break
# 시간은 n 분이내 동영상만 올려주세요 문구를 통해 조정

file = 'C:/Users/user/Downloads/123.mp4'
newmap = 'C:/Users/user/Downloads/test'

#try:
#    if not os.path.exists(filename[:-4]):
#        os.makedirs(filename[:-4])
#except OSError:
#    print ('Error: Creating directory. ' +  filename[:-4])


#portal = filename[:-4]

frame(file,newmap)


########################################################

def gradient(a): # 기울기
    g=(a[3]-a[1])/(a[2]-a[0])   # delta y/ delta x
    return g


def analcoc(chaseon):
    count=0
    for j in chaseon:
        for i in j:
            if i==[]:
                continue
            a=gradient(i)
            if abs(a)>math.tan(1.48): # 차선의 기울기가 85도를 넘어가면
                count+=1
                if count==5: # 5프레임 이상 지속되면
                    return 'yes' # 차선 변경이 있었다
    return 'no'  # 차선 변경이 없었다

############## 파이썬