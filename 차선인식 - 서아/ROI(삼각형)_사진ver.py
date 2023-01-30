import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2


# Read in the image and print some stats
image = mpimg.imread('C:/Users/user/Downloads/test/1.png')

image = cv2.resize(image, (1280,720))
print('This image is: ', type(image), 
         'with dimensions:', image.shape)

# Pull out the x and y sizes and make a copy of the image
ysize = image.shape[0]
xsize = image.shape[1]
region_select = np.copy(image)

# 삼각형 모양으로 roi 지정하기
# 아래의 세가지 변수를 바꿔가며 삼각형 모양안에 차선이 다들어오도록 만들어 줍니다.
left_bottom = [0, 720]
right_bottom = [1280, 720]
apex = [640, 0]

# np.polyfit()함수는 괄호안에 x좌표, y좌표, 차수를 넣으면 두 점을 연결한 직선의 기울기와 y절편을 알려줍니다.
fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# 위에서 그린 선 안에 있는 모든 픽셀을 선택합니다.
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))

# 위에서 선택한 픽셀의 색을 빨간색으로 만든다. 즉, roi부분을 빨간색으로 만든다.
region_select[region_thresholds] = [0, 255, 0]

# Display the image
plt.imshow(region_select)

# 그래프가 나타나지않으면 아래의 주석을 풀면됩니다.
plt.show()

######################################################################################################

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Read in the image
image = mpimg.imread('C:/Users/user/Downloads/test/1.png')

image = cv2.resize(image, (1280,720))

# Grab the x and y sizes and make two copies of the image
# With one copy we'll extract only the pixels that meet our selection,
# then we'll paint those pixels red in the original image to see our selection 
# overlaid on the original.
ysize = image.shape[0]
xsize = image.shape[1]
color_select= np.copy(image)
line_image = np.copy(image)

# Define our color criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

# Define a triangle region of interest (Note: if you run this code, 
# Keep in mind the origin (x=0, y=0) is in the upper left in image processing
# you'll find these are not sensible values!!
# But you'll get a chance to play with them soon in a quiz ;)
left_bottom = [0, 720]
right_bottom = [1280, 720]
apex = [640, 0]

fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

# Mask pixels below the threshold
color_thresholds = (image[:,:,0] < rgb_threshold[0]) | \
                    (image[:,:,1] < rgb_threshold[1]) | \
                    (image[:,:,2] < rgb_threshold[2])

# Find the region inside the lines
XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
region_thresholds = (YY > (XX*fit_left[0] + fit_left[1])) & \
                    (YY > (XX*fit_right[0] + fit_right[1])) & \
                    (YY < (XX*fit_bottom[0] + fit_bottom[1]))
# Mask color selection
color_select[color_thresholds] = [0,0,0]
# Find where image is both colored right and in the region
line_image[~color_thresholds & region_thresholds] = [0,255,0]

# Display our two output images
plt.imshow(color_select)
plt.imshow((line_image * 255).astype(np.uint8))

# uncomment if plot does not display
# plt.show()