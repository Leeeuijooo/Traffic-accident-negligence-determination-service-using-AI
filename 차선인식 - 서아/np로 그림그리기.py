import cv2
import numpy as np
import math

#np.zeros((512, 512, 3), np.uint8) -> 모두 0으로 된 빈 Black canvas
# 512×512 크기의 3개의 채널로 되어 있으며 채널의 구성 데이터 타입은 unit8
#img = np.zeros((512, 512, 3), np.uint8) # np.uint8 -> 양수만 표현 가능, 2^8개수 만큼 표현 가능(0~255)
img = np.zeros((512, 512, 3), np.uint8) + 255
# 255를 더하면 흰색 배경

blue_color = (255,0,0)
green_color = (0, 255, 0)
red_color = (0, 0, 255)
white_color = (255, 255, 255)




img = cv2.line(img, (0, 0), (511, 511), green_color, 10) # -> (0(B), 0(G), 255(R))
 #img = cv2.line(img, (왼쪽상단 (위), 왼쪽상단 (아래)), (오른 쪽 하단 기울기, 선 길이), color=(B, G, R), 선 굵기)
    #cv2.line(img, start, end, color, thickness)

img = cv2.line(img, (150,150), (150,150), red_color, 10)
img = cv2.line(img, (256,500), (256,500), blue_color, 15) # 시작점 

a = -50
b = 50

img = cv2.line(img, (256-a,500-b), (256-a,500-b), blue_color, 15) # 시작점 




##########################################################################
##########################################################################
def gradient(a): # 기울기
    g=(a[3]-a[1])/(a[2]-a[0])   # delta y/ delta x
    return g

def check(chaseon):
    count=0
    a=0
    b=0
    for j in chaseon:
        for i in j:
            if i==[]:
                continue
            a=gradient(i)
            if abs(a)>math.tan(1.48): # 차선의 기울기가 85도를 넘어가면
                count+=1
                if count==5: # 5프레임 이상 지속되면
                    a = -50
                    b = 50  # 차선 변경이 있었다
                
            else:
                a = 0
                b = 0 # 차선 변경이 없었다
    return a,b 

def region_of_interest(img, vertices, color3=(255,255,255), color1=255): # ROI 셋팅

    mask = np.zeros_like(img) # mask = img와 같은 크기의 빈 이미지

    if len(img.shape) > 2: # Color 이미지(3채널)라면 :
        color = color3
    else: # 흑백 이미지(1채널)라면 :
        color = color1

    # vertices에 정한 점들로 이뤄진 다각형부분(ROI 설정부분)을 color로 채움
    for vertices in vertices:
        if len(vertices) > 0:
            mask = cv2.fillPoly(mask, [vertices], color)

    
    # cv2.fillPoly(mask, vertices, color)

    # 이미지와 color로 채워진 ROI를 합침
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image




def gaussian_blur(img, kernel_size): # 가우시안 필터
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def doit(newmap,frcount,ROI_range):
    chaseon=[]
    for i in range(1,frcount+1):
        image = cv2.imread(newmap+"/"+str(i)+'.png') # 이미지 읽기
        height, width = image.shape[:2] # 이미지 높이, 너비

        cv2.imshow('img',image)

        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # 흑백이미지로 변환
        # 아니면 cv2.imread('~~.jpg', cv2.IMREAD_GRAYSCALE) 이용해서 바로 읽어올 수도 있음

        blur_img = gaussian_blur(gray_img, 3) # Blur 효과

        canny_img = cv2.Canny(blur_img, 50, 160) # Canny edge 알고리즘
        cv2.imshow('canny',canny_img)



        # vertices : ROI 구역으로 할 좌표 점
        # np.int32 정수형 / np.float64 실수형
        # region_of_interest -> ROI 구역 만드는 함수
        vertices = np.array([ROI_range], dtype=np.int32)
        ROI_img = region_of_interest(canny_img, vertices)

        # np.squeeze : 크기가 1인 axis 제거
        line_arr = cv2.HoughLines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환   #[x1, y1, x2, y2]
        line_arr = np.squeeze(line_arr)
        
        print(line_arr)

        # 기울기 구하기
        # np.arctan2 : 출력 범위가 [-pi, pi] -> 일반 arctan는 180도 이상 차이나는 값 구분 X 때문
        slope_degree = (np.arctan2(line_arr[:,1] - line_arr[:,3], line_arr[:,0] - line_arr[:,2]) * 180) / np.pi

        # 수평 기울기 제한
        line_arr = line_arr[np.abs(slope_degree)<160]
        slope_degree = slope_degree[np.abs(slope_degree)<160]
        # 수직 기울기 제한
        line_arr = line_arr[np.abs(slope_degree)>95]
        slope_degree = slope_degree[np.abs(slope_degree)>95]
        # 필터링된 직선 버리기
        L_lines, R_lines = line_arr[(slope_degree>0),:], line_arr[(slope_degree<0),:]
        L_lines, R_lines = L_lines[:,None], R_lines[:,None]
        for j in L_lines:
            for k in j:
                for l in k:
                    l=int(l)
        for j in R_lines:
            for k in j:
                for l in k:
                    l=int(l)
        temp1=[list(j) for j in L_lines]+[list(j) for j in R_lines]
        rrealtemp=[]
        temp2=[]
        for j in temp1:
            for k in j:
                for l in k:
                    rrealtemp.append(int(l))
                temp2.append(rrealtemp)
                rrealtemp=[]
        chaseon.append(temp2)
    return chaseon  #3차

def hoit(search_range,newmap,av,inf):
    left_count=0
    right_count=0
    av=SAM_572(search_range,newmap,av,1)
    start=av[1]  #처음 내 앞의 상
    for i in range(2,inf[0]+1):
        av=SAM_572(search_range,newmap,av,i)
        if sub(av[i][0][0],start[0][0])+sub(av[i][0][1],start[0][1])+sub(av[i][0][2],start[0][2])>=40: #BGR 값의 변화
            left_count+=1 
        if sub(av[i][1][0],start[1][0])+sub(av[i][1][1],start[1][1])+sub(av[i][1][2],start[1][2])>=40: #BGR값의 변화
            right_count+=1
        if sub(av[i][0][0],start[0][0])+sub(av[i][0][1],start[0][1])+sub(av[i][0][2],start[0][2])<40 and left_count!=0:
            left_count=0
        if sub(av[i][1][0],start[1][0])+sub(av[i][1][1],start[1][1])+sub(av[i][1][2],start[1][2])<40 and right_count!=0:
            right_count=0
        if left_count>=3 and right_count>=3:
            return ('no',i)
        if left_count==5:
            return ('left',i)
        if right_count==5:
            return ('right',i)
    return ('no',0)

def sub(a,b):
    if a>=b:
        return int(a-b)
    return int(b-a)

def SAM_572(search_range,newmap,aV,a):  #search_range=[[a1,b1],[a2,b1],[a3,b2],[a4,b2],[기울기, y절편],[기울기, y절편]]
    av=aV
    temp1=[0,0,0]  #왼쪽
    temp2=[0,0,0]  #오른쪽
    temp=[]
    left=search_range[4]
    right=search_range[5]
    img=cv2.imread(newmap+"/"+str(a)+".png")  #이미지 불러오기
    for k in range(search_range[0][1],search_range[2][1]): #각각의 행
        for j in range(int((k-left[1])/left[0]),int(((k-left[1])/left[0]+int(k-right[1])/right[0])//2)):#왼쪽 부분
            for h in range(3):
                temp1[h]+=int(img[k][j][h]) #BGR값 모두 합하기
        for j in range(int(((k-left[1])/left[0]+(k-right[1])/right[0])/2),int((k-right[1])/right[0]+1)): #오른쪽 부분
            for h in range(3):
                temp2[h]+=int(img[k][j][h]) #BGR값 모두 합하기
    for h in range(3):
        temp1[h]=temp1[h]//((search_range[1][0]-search_range[0][0]+search_range[3][0]-search_range[2][0])*(search_range[2][1]-search_range[0][1])//4)  #평균 내기
        temp2[h]=temp2[h]//((search_range[1][0]-search_range[0][0]+search_range[3][0]-search_range[2][0])*(search_range[2][1]-search_range[0][1])//4)
    temp.append(temp1)
    temp.append(temp2)
    av.append(temp)
    return av

def makesr(chaseon,ROI_range,inf):
    search_range=[0,0,0,0]
    left=[0,0] #기울기, y절편   #왼쪽 차선
    right=[0,0] #오른쪽 차선
    left[0]=gradient(chaseon[0][0])  #왼쪽 기울기
    right[0]=gradient(chaseon[0][-1])  #오른쪽 기울기
    left[1]=chaseon[0][0][1]-left[0]*chaseon[0][0][0]
    right[1]=chaseon[0][-1][1]-right[0]*chaseon[0][-1][0]
    b1=ROI_range[0][1]+inf[3]//12  #도로인 첫 y지점에서 적당히 밑으로
    a1=(b1-left[1])/left[0]  #왼쪽 차선에 접하게
    a2=(b1-right[1])/right[0]  #오른쪽 차선에 접하게
    b2=ROI_range[3][1]-inf[3]//12   #도로인 마지막 y지점에서 적당히 위
    a3=(b2-left[1])/left[0]  #왼쪽 차선에 접하게
    a4=(b2-right[1])/right[0]  #오른쪽 차선에 접하게
    
    
    search_range[0]=[a1,b1]
    search_range[1]=[a2,b1]
    search_range[2]=[a3,b2]
    search_range[3]=[a4,b2]
    search_range.append(left)
    search_range.append(right)
    
    road=[0,0,0,0]
    y2=ROI_range[3][1]-inf[3]//30
    y1=ROI_range[3][1]-inf[3]//15
    x1=(y1-left[1])/left[0]+20  #왼쪽 차선에 접하게
    x2=(y1-right[1])/right[0]-20  #오른쪽 차선에 접하게
    x3=(y2-left[1])/left[0]+20  #왼쪽 차선에 접하게
    x4=(y2-right[1])/right[0]-20  #오른쪽 차선에 접하게
    road[0]=[x1,y1]
    road[1]=[x2,y1]
    road[2]=[x3,y2]
    road[3]=[x4,y2]
    
    return search_range,road

def info(filename): #[총 프레임 수, fps, 가로, 세로 길이] 리턴
    vid=cv2.VideoCapture(filename)
    temp=[]
    temp.append(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))) #총 프레임 수
    temp.append(int(vid.get(cv2.CAP_PROP_FPS))) #fps
    temp.append(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))) #가로
    temp.append(int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))) #세로
    return temp

file = 'C:/Users/user/Downloads/123.mp4'
newmap = 'C:/Users/user/Downloads/test_a'

inf=info(file)

ROI_range=[]
img=cv2.imread(newmap+"/1.png")
img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)

while 1:
	cv2.imshow('ROI range',img)
	k=cv2.waitKey(1) & 0xFF
	if k==27: break #esc키를 누르면 종료
cv2.destroyAllWindows()
#ROI_range=[ROI_range[0],[0,0],[0,0],ROI_range[1]]
#ROI_range[1]=[ROI_range[3][0],ROI_range[0][1]]
#ROI_range[2]=[ROI_range[0][0],ROI_range[3][1]]

ROI =  [[128, 310], [1136, 310], [128, 582], [1136, 582]]

av=['sam']
tempchaseon=doit(newmap,5,ROI_range)
search_range,road=makesr(tempchaseon,ROI_range,inf)  #내 차선에 내접하는 사다리꼴 만들
direction,frcount=hoit(search_range,newmap,av,inf)  #두 차의 위치관계 #left or right
chaseon=doit(newmap,frcount,ROI_range)

a,b=check(chaseon)  

cv2.imshow('image', img)# image라는 이름으로 불러온다.
cv2.waitKey(0) 
# waitKey(0): 무한대기
cv2.destroyAllWindows()





