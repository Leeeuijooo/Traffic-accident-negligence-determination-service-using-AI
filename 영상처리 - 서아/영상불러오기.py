# 영상받아오기 test

import cv2
import os
 
path = 'C:/Users/user/Downloads'
filePath = os.path.join(path, "123.mp4")  # 아마 컴퓨터 내 경로 지정해서 가져오는 듯
print(filePath)

if os.path.isfile(filePath):	# 해당 파일이 있는지 확인
    # os.path.isfile() : 파일 유무 판단
    # 파일이면 True, 아니면 False, 없어도 False 반환

    # 영상 객체(파일) 가져오기
    cap = cv2.VideoCapture(filePath)
else:
    print("파일이 존재하지 않습니다.")  

# 프레임을 정수형으로 형 변환
frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))	# 영상의 넓이(가로) 프레임
frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))	# 영상의 높이(세로) 프레임

# CAP_PROP_FPS: 초당 프레임의 수
# CAP_PROP_ZOOM: 카메라 줌
# CAP_PROP_FRAME_COUNT : 비디오 파일의 총 프레임 수
# CAP_PROP_POS_MSEC : 밀리초 단위로 현재 위치
# CAP_PROP_POS_FRAME : 현재 프레임 번호
# CAP_PROP_EXPOSURE : 노출

# 각 항목 확인할 떄는 'get', 변경할 때는 'set'
# cap.get(cv2.CAP_PROP_FRAME_WIDTH) >> 프레임 폭 반환
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320) >> 프레임의 폭 320으로 변경


frame_size = (frameWidth, frameHeight)
print('frame_size={}'.format(frame_size))

frameRate = 33   # FPS를 말하는 듯
 
while True:
    # 한 장의 이미지(frame)를 가져오기
    # 영상 : 이미지(프레임)의 연속
    # 정상적으로 읽어왔는지 -> retval
    # 읽어온 프레임 -> frame
    retval, frame = cap.read()
    if not(retval):	# 프레임정보를 정상적으로 읽지 못하면
        break  # while문을 빠져나가기
        
    cv2.imshow('frame', frame)	# 프레임 보여주기
    key = cv2.waitKey(frameRate)  # frameRate msec동안 한 프레임을 보여준다
    
    # 키 입력을 받으면 키값을 key로 저장 -> Esc == 27(아스키코드)
    # Esc 키 누르면 종료됨
    if key == 27:
        break	# while문을 빠져나가기
        
if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
    cap.release()	# 영상 파일(카메라) 사용을 종료
    
cv2.destroyAllWindows()
