#
# 차선인식 ver2 + 사진 ver 이용해 v-roi 생성해보기

import cv2
import copy
import sys
import math
import cv2 as cv
import numpy as np

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

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def mark_img(img, blue_threshold=200, green_threshold=200, red_threshold=200): # 흰색 차선 찾기

    #  BGR 제한 값
    bgr_threshold = [blue_threshold, green_threshold, red_threshold]

    # BGR 제한 값보다 작으면 검은색으로
    thresholds = (img[:,:,0] < bgr_threshold[0]) \
                | (img[:,:,1] < bgr_threshold[1]) \
                | (img[:,:,2] < bgr_threshold[2])
    mark[thresholds] = [0,0,0]
    return mark

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


cap = cv2.VideoCapture("C:/Users/user/Downloads/123.mp4") # 동영상 불러오기

while(cap.isOpened()):
    ret, img = cap.read()

    if ret == True :
        img = cv2.resize(img, (1280, 720))
        
        height, width = img.shape[:2] # 이미지 높이, 너비

        # 사다리꼴 모형의 Points
        imshape = img.shape
        gray = grayscale(img)
        
        kernel_size = 5
        blur_gray = gaussian_blur(gray, kernel_size)
        
        
        vertices_l = np.array([[(100,imshape[0]),
                     (300, 350),
                     (600, 350),
                     (400, imshape[0])]], dtype=np.int32)

        vertices_r = np.array([[(imshape[1]-400,imshape[0]),
                     (imshape[1]-600, 350),
                     (imshape[1]-300, 350),
                     (imshape[1]-100, imshape[0])]], dtype=np.int32)
        
        low_threshold = 50
        high_threshold = 200
        edges = cv2.Canny(blur_gray, low_threshold, high_threshold)


        mask_l = region_of_interest(edges, vertices_l)
        mask_r = region_of_interest(edges, vertices_r)

        mask = mask_l + mask_r
        
        verti = vertices_l + vertices_r
        roi_img_l = region_of_interest(img, vertices_l, (0,0,255)) 
        roi_img_r = region_of_interest(img, vertices_r, (0,0,255)) 
        
        roi_img = roi_img_l + roi_img_r

        mark = np.copy(roi_img) # roi_img 복사
        mark = mark_img(roi_img) # 흰색 차선 찾기

        # 흰색 차선 검출한 부분을 원본 image에 overlap 하기
        color_thresholds = (mark[:,:,0] == 0) & (mark[:,:,1] == 0) & (mark[:,:,2] > 200)
        img[color_thresholds] = [0,0,255]

        cv2.imshow('results',img) # 이미지 출력
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    else :
        break
    
# Release
cap.release()
cv2.destroyAllWindows()