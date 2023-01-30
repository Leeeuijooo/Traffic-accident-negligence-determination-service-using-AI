# image_detection

## 차선인식 알고리즘 연습
# https://pinkwink.kr/1264 참고

from array import ArrayType
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#img = mping.imread('C:/Users/user/Downloads/146.png') 사용하면 색 제대로 출력
img = cv2.imread('C:/Users/user/Downloads/test_a/10.jpg')
# 파일 경로 써주기
#plt.imshow('img',img)

# printing out some stats and plotting the image
print('This image is:', type(img), 'with dimensions:', img.shape)
plt.figure(figsize=(10,8))
plt.imshow(img)
plt.show()

## grayscale 진행

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

gray = grayscale(img)
plt.figure(figsize=(10,8))
plt.imshow(gray, cmap='gray')
plt.show()
print('grayscale 적용 후')


# 가우시안 블러 처리

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

kernel_size = 5
blur_gray = gaussian_blur(gray, kernel_size)

plt.figure(figsize=(10,8))
plt.imshow(blur_gray, cmap='gray')
plt.show()
print('blur 적용 후')

## 케니 엣지 적용

#def canny(img, low_threshold, high_threshold) :
#    """Applies the Canny transform"""
#    return cv2.Canny(img, low_threshold, high_threshold)


#blur_gray=cv2.imread('blur_gray')

low_threshold = 50
high_threshold = 200
edges = cv2.Canny(blur_gray, low_threshold, high_threshold)

plt.figure(figsize=(10,8))
plt.imshow(edges, cmap='gray')
plt.show()
print('canny 적용 후')

# 관심 영역 재설정

# 이미지와 같은 크기의 검정색 채워주기
import numpy as np

mask = np.zeros_like(img)

plt.figure(figsize=(10,8))
plt.imshow(mask, cmap='gray')
plt.show()

# 관심을 가질만한 영역 잡아주기

# 밑은 테스트
#if len(img.shape) > 2:
#    channel_count = img.shape[2] # 이미지에 따라 3 or 4 ?
#    ignore_mask_color = (255,) * channel_count
#else:
#    ignore_mask_color = 255
    
#imshape = img.shape
#print(imshape)

#vertices = np.array([[(100,imshape[0]),
#                    (450, 320),
#                    (550, 320),
#                   (imshape[1]-20, imshape[0])]], dtype=np.int32)

#cv2.fillPoly(mask, vertices, ignore_mask_color)

#plt.figure(figsize=(10,8))
#plt.imshow(mask, cmap='gray')
#plt.show()

def region_of_interest(img, vertices):
    mask=np.zeros_like(img)
    
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else :
        ignore_mask_color = 255
        
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

imshape = img.shape
vertices_l = np.array([[(100,imshape[0]),
                     (300, 350),
                     (600, 350),
                     (400, imshape[0])]], dtype=np.int32)

vertices_r = np.array([[(imshape[1]-400,imshape[0]),
                     (imshape[1]-600, 350),
                     (imshape[1]-300, 350),
                     (imshape[1]-100, imshape[0])]], dtype=np.int32)



mask_l = region_of_interest(edges, vertices_l)
mask_r = region_of_interest(edges, vertices_r)

mask = mask_l + mask_r

plt.figure(figsize=(10,8))
plt.imshow(mask, cmap='gray')
plt.show()


# 직접 선을 그리는 함수 : 허프라인 사용

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

rho=2
theta = np.pi/180
threshold = 90
min_line_len = 120
max_line_gap = 150

lines = hough_lines(mask, rho, theta, threshold,
                    min_line_len, max_line_gap)

plt.figure(figsize=(10,8))
plt.imshow(lines, cmap='gray')
plt.show()


# 원본 사진에 덮기

def weighted_img(img, initial_img, alpha=0.8, beta=1., gamma=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, gamma)

lines_edges = weighted_img(lines, img, alpha=0.8, beta=1., gamma=0.)

plt.figure(figsize=(10,8))
plt.imshow(lines_edges)
plt.show()


