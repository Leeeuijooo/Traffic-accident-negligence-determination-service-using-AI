# 차선인식 excode2 연계


import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import numpy as np

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
        
def info(filename): #[총 프레임 수, fps, 가로, 세로 길이] 리턴
    vid=cv2.VideoCapture(filename)
    temp=[]
    temp.append(int(vid.get(cv2.CAP_PROP_FRAME_COUNT))) #총 프레임 수
    temp.append(int(vid.get(cv2.CAP_PROP_FPS))) #fps
    temp.append(int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))) #가로
    temp.append(int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))) #세로
    return temp


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask=np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else :
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    ROI_image = cv2.bitwise_and(img, mask)
    return ROI_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
            
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# 원본 사진에 덮기
def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)


def sub(a,b):    # 단순한 빼기(-) 함수
    if a>=b:
        return int(a-b)
    return int(b-a)



def doit(newmap,frcount,ROI_range):
    chaseon=[]
    for i in range(1,frcount+1):
        image = cv2.imread(newmap+"/"+str(i)+'.png') # 이미지 읽기
        height, width = image.shape[:2] # 이미지 높이, 너비


        gray_img = grayscale(image) # 흑백이미지로 변환
        # 아니면 cv2.imread('~~.jpg', cv2.IMREAD_GRAYSCALE) 이용해서 바로 읽어올 수도 있음

        blur_img = gaussian_blur(gray_img, 3) # Blur 효과

        canny_img = cv2.Canny(blur_img, 50, 160) # Canny edge 알고리즘

        # vertices : ROI 구역으로 할 좌표 점
        # np.int32 정수형 / np.float64 실수형
        # region_of_interest -> ROI 구역 만드는 함수
        vertices = np.array([ROI_range], dtype=np.int32)
        ROI_img = region_of_interest(canny_img, vertices)

        # np.squeeze : 크기가 1인 axis 제거
        line_arr = hough_lines(ROI_img, 1, 1 * np.pi/180, 30, 10, 20) # 허프 변환   #[x1, y1, x2, y2]
        line_arr = np.squeeze(line_arr)

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


    # 영상처리에서 delta는 영상 기본 밝기 값으로 전체 픽셀 값 조정
    # cv2.imshow("delta0",delta0) 
    # 이 delta가 아닌 것 같고 다른 기울기 인듯 ?
    
def gradient(a): # 기울기
    g=(a[3]-a[1])/(a[2]-a[0])   # delta y/ delta x
    return g


def judge_gradient(chaseon):
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

def SAM_572(search_range,newmap,aV,a):  #search_range=[[a1,b1],[a2,b1],[a3,b2],[a4,b2],[기울기, y절편],[기울기, y절편]]
    av=aV
    temp1=[0,0,0]  # 왼쪽
    temp2=[0,0,0]  # 오른쪽
    temp=[]
    left=search_range[4]   # [기울기, y절편]
    right=search_range[5]   # [기울기, y절편] 
    img=cv2.imread(newmap+"/"+str(a)+".png")  # 이미지 불러오기
    for k in range(search_range[0][1],search_range[2][1]): # 각각의 행
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

def hoit(search_range,newmap,av,inf):
    left_count=0
    right_count=0
    av=SAM_572(search_range,newmap,av,1)
    start=av[1]  # 처음 내 앞의 상
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

def makesr(chaseon,ROI_range,inf):
    search_range=[0,0,0,0]
    left=[0,0] # 기울기, y절편   # 왼쪽 차선
    right=[0,0] # 오른쪽 차선
    left[0]=gradient(chaseon[0][0])  # 왼쪽 기울기
    right[0]=gradient(chaseon[0][-1])  # 오른쪽 기울기
    left[1]=chaseon[0][0][1]-left[0]*chaseon[0][0][0]
    right[1]=chaseon[0][-1][1]-right[0]*chaseon[0][-1][0]
    b1=ROI_range[0][1]+inf[3]//12  # 도로인 첫 y지점에서 적당히 밑으로
    a1=(b1-left[1])/left[0]  # 왼쪽 차선에 접하게
    a2=(b1-right[1])/right[0]  # 오른쪽 차선에 접하게
    b2=ROI_range[3][1]-inf[3]//12   # 도로인 마지막 y지점에서 적당히 위
    a3=(b2-left[1])/left[0]  # 왼쪽 차선에 접하게
    a4=(b2-right[1])/right[0]  # 오른쪽 차선에 접하게
    
    
    search_range[0]=[a1,b1]
    search_range[1]=[a2,b1]
    search_range[2]=[a3,b2]
    search_range[3]=[a4,b2]
    search_range.append(left)
    search_range.append(right)
    
    road=[0,0,0,0]
    y2=ROI_range[3][1]-inf[3]//30
    y1=ROI_range[3][1]-inf[3]//15
    x1=(y1-left[1])/left[0]+20  # 왼쪽 차선에 접하게
    x2=(y1-right[1])/right[0]-20  # 오른쪽 차선에 접하게
    x3=(y2-left[1])/left[0]+20  # 왼쪽 차선에 접하게
    x4=(y2-right[1])/right[0]-20  # 오른쪽 차선에 접하게
    road[0]=[x1,y1]
    road[1]=[x2,y1]
    road[2]=[x3,y2]
    road[3]=[x4,y2]
    
    return (search_range,road)

def roadcolor(newmap,road):
    cnt=1
    road_color=[0,0,0]
    while True:
        img=cv2.imread(newmap+'/'+str(cnt)+'.png')
        for i in range(road[2][0]+5,road[3][0]-5):
            road_color[0]+=img[i][road[2][1]][0]
            road_color[1]+=img[i][road[2][1]][1]
            road_color[2]+=img[i][road[2][1]][2]
        road_color[0]=road_color[0]//(road[3][0]-road[2][0])
        road_color[1]=road_color[0]//(road[3][0]-road[2][0])
        road_color[2]=road_color[0]//(road[3][0]-road[2][0])
        if road_color[0]<150 and road_color[1]<150 and road_color[2]<150:
            return road_color
        cnt+=1
        
        
        
        
def draw_rec(event, x, y, flags, param): #이 함수는 모듈화 시 오류가 발생하여 main.py에 넣었습니다.
	global ix, iy, drawing, mode, search_range, img
	if event==cv2.EVENT_LBUTTONDOWN: #마우스 좌클릭이 감지되면
		ix,iy=x,y
		if len(search_range)==0:
			search_range.append([x*2,y*2])
		elif [x,y]!=search_range[-1]:
			search_range.append([x*2,y*2]) #마우스의 x,y좌표값을 search_range에 append
	if len(search_range)==2:
		cv2.rectangle(img,tuple([search_range[0][0]//2,search_range[0][1]//2]),tuple([search_range[1][0]//2,search_range[1][1]//2]),(0,255,0),-1) #탐색에 필요한 두 기준점이 잡히면 그것으로 직사각형 그림
        # cv2.rectangle(이미지, 시작점 좌표(x,y), 종료점 좌표(x,y), 색상[, 두께[, shift]])
        # https://copycoding.tistory.com/146 (예시)
        
def draw_rec2(event, x, y, flags, param): #이 함수는 모듈화 시 오류가 발생하여 main.py에 넣었습니다.
	global ix, iy, drawing, mode, ROI_range, img
	if event==cv2.EVENT_LBUTTONDOWN: #마우스 좌클릭이 감지되면
		ix,iy=x,y
		if len(ROI_range)==0:
			ROI_range.append([x*2,y*2])
		elif [x,y]!=ROI_range[-1]:
			ROI_range.append([x*2,y*2]) #마우스의 x,y좌표값을 ROI_range에 append
	if len(ROI_range)==2:
		cv2.rectangle(img,tuple([ROI_range[0][0]//2,ROI_range[0][1]//2]),tuple([ROI_range[1][0]//2,ROI_range[1][1]//2]),(0,255,0),-1) #탐색에 필요한 두 기준점이 잡히면 그것으로 직사각형 그림
#img = np.zeros((720, 1280, 3), np.uint8)
        
#########################
file= 'C:/Users/user/Downloads/123.mp4'# 영상 경로
newmap= 'C:/Users/user/Downloads/acc'# 파일을 저장할 경로
inf=info(file)

frame(file,newmap)
##########################



ROI_range=[]
img=cv2.imread(newmap+"/1.png")
img=cv2.resize(img,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
cv2.namedWindow('ROI range')
cv2.setMouseCallback('ROI range',draw_rec2)
print('마우스로 영상을 탐색할 두 기준점을 찍은 후, 직사각형이 그려지면 esc키를 누르세요.')
while 1:
	cv2.imshow('ROI range',img)
	k=cv2.waitKey(1) & 0xFF
	if k==27: break #esc키를 누르면 종료
cv2.destroyAllWindows()

ROI_range=[ROI_range[0],[0,0],[0,0],ROI_range[1]]
ROI_range[1]=[ROI_range[3][0],ROI_range[0][1]]
ROI_range[2]=[ROI_range[0][0],ROI_range[3][1]]
###############영상 가공 및 값 추출#################

av=['sam']
tempchaseon=doit(newmap,5,ROI_range)  #5프레임 정도 미리 차선 따놓기

search_range=makesr(tempchaseon,ROI_range,inf)  #내 차선에 내접하는 사다리꼴 만들

direction,frcount=hoit(search_range,newmap,av,inf)  #두 차의 위치관계 #left or right
img=cv2.imread(newmap+"/1.png")

chaseon=doit(newmap,frcount,ROI_range)  #차선 리스트

cs=judge_gradient(chaseon)      #내가 차선 변경 하였는지 #yes or no












