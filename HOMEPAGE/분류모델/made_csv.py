import main
import pandas as pd
import numpy as np
import cv2
import os


detect_list = []


def check(file, newmap, ROI_range):
    inf = main.info(file)
    #print('inf : ',inf)
    #print('ROI = ', ROI_range)
    av = ['sam']
    tempchaseon = main.doit(newmap, 3, ROI_range)  # 5프레임 정도 미리 차선 따놓기
    # print('tempchaseon=',tempchaseon)

    if tempchaseon == None:
        return 0

    else:
        search_range = main.makesr(tempchaseon, ROI_range, inf)
        direction, frcount = main.hoit(search_range, newmap, av, num)
        img = cv2.imread(newmap+"/1.png")
        # print('direction,frcount=',direction,frcount)
        chaseon = main.doit(newmap, num-4, ROI_range)
        # print(len(chaseon), len(chaseon[0]), len(chaseon[0][0]))
        # print('chaseon=',chaseon)
        cs = main.rl_detec(chaseon)
        return cs


for i in range(2, 3):
    data = []
    try:
        data.append(i)
        path = 'C:/Users/dnjsw/Desktop/video'
        filePath = os.path.join(path, str(i)+'.mp4')
        file = filePath

        newmap = 'C:/Users/dnjsw/Desktop/test_b'

        inf1 = main.info(file)

        num = main.frame_2(file, newmap, inf1)
        print('num: ', num)

        ROI_range = []
        search_range = []
        img = cv2.imread(newmap+"/1.png")
        img = cv2.resize(img, (1280, 720))
        # cv2.INTER_AREA : 영상 축소

        if len(ROI_range) == 0:
            ROI_range.append([400, 400])   # 왼쪽 위에 꼭짓점
            ROI_range.append([1000, 700])  # 오른쪽 아래 꼭짓점
        if len(ROI_range) == 2:
            cv2.rectangle(img, tuple([ROI_range[0][0]//2, ROI_range[0][1]//2]), tuple(
                [ROI_range[1][0]//2, ROI_range[1][1]//2]), (0, 255, 0), -1)
        # 탐색에 필요한 두 기준점이 잡히면 그것으로 t사다리꼴 생성

        ROI_range = [ROI_range[0], [0, 0], [0, 0], ROI_range[1]]
        ROI_range[1] = [600, 400]
        ROI_range[2] = [200, 700]

        print(i, '번째 데이터')

# 0 : 부정적 / 1 : 긍정적
# data = [동영상 넘버, 차량 검출 가능 여부, 차선변경 여부, 우회전, 좌회전]
        if check(file, newmap, ROI_range) == 0:
            print('차선 검출 실패 : 차선이 너무 복잡하거나 흐릿합니다.')
            detection = 0
            data.append(detection)
            data.append(0)
            data.append(0)
            data.append(0)

        elif check(file, newmap, ROI_range) != 0:
            result_cs = check(file, newmap, ROI_range)
            print('차선 변경 : ', result_cs)
            detection = 1
            data.append(detection)

            if result_cs == '변경 안함':
                lane_change = 0
                data.append(lane_change)
                data.append(0)
                data.append(0)
            elif result_cs == '차선 변경':
                lane_change = 1
                data.append(lane_change)
                data.append(0)
                data.append(0)
            elif result_cs == '오른쪽 차선 변경':
                right = 1
                data.append(0)
                data.append(right)
                data.append(0)
            elif result_cs == '왼쪽 차선 변경':
                left = 1
                data.append(0)
                data.append(0)
                data.append(left)

        print(data)

        detect_list.append(data)
    except:
        print("예외 종료")
        continue


print(detect_list)

# df = pd.DataFrame(detect_list, columns=['video_num','detection','lane_change','right','left'])
# df.to_csv("detect_list",  encoding="utf-8-sig")
