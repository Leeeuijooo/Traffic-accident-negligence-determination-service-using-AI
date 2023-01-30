from doctest import master
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from yolov5 import detect
import os
import pandas as pd
from . import trans
from . import result


def home(request):
    return render(request,
                  'home/home.html')


def check(request):
    if 'file1' in request.FILES:
        file = request.FILES['file1']

        fs = FileSystemStorage()
        fs.save("blackbox.mp4", file)

    else:
        return render(request, 'check/check.html')
    return render(request, 'check/check.html')


def detectobj(request):

    if 'file' in request.FILES:

        glist = request.POST.getlist("chk[]")
        print("glist: ", glist)

        # 객체탐지 실행
        detect.run()

        # txt파일 저장
        # directory = "C:/Users/user/Desktop/hanlab/22_hf246/HOMEPAGE/prj/result/detectedvideo/video/labels"
        directory = "C:/Users/user/Desktop/hanlab/22_hf246/HOMEPAGE/prj/results/detectedvideo/labels"
        outfile_name = "merge.txt"
        out_file = open(outfile_name, 'w')
        input_files = os.listdir(directory)

        for filename in input_files:
            if ".txt" not in filename:
                continue
            file = open(directory + "/" + filename)
            content = file.read()
            out_file.write(content)
            file.close()
        out_file.close()

        read_file = pd.read_csv(
            r'C:/Users/user/Desktop/hanlab/22_hf246/HOMEPAGE/prj/merge.txt')
        # read_file.to_csv(
        #    r'C:/Users/user/Desktop/hanlab/22_hf246/HOMEPAGE/prj/merge.csv', index=None)

        file_path = "C:/Users/user/Desktop/hanlab/22_hf246/HOMEPAGE/prj/merge.txt"

        res = []
        with open(file_path) as f:
            lines = f.read().splitlines()

        res = []
        for i in range(0, len(lines)-1):
            a = ''.join(s for s in lines[i])
            b = a.split()

            c = b[0]
            res.append(c)
    
        result_value = list(set(res))
        
        final_list = glist+result_value
        print("final_list: ", final_list)
        
        critical_factor=trans.make(final_list)
        print("critical_factor: ",critical_factor)
        critical_index = ['상대방차 차선 변경',
        '내차 차선변경',
        '빨간불',
        '좌측깜박이',
        '초록불',
        '역삼각형',
        '합류도로 노면표시',
        '중앙분리대',
        '속도100',
        '유도봉',
        '속도50',
        '좌회전신호',
        '속도110',
        '우측깜박이',
        '회전교차로',
        '동시신호',
        '직진 후 좌회전',
        '노란불',
        '유턴',
        '감소도로 화살표']

        results = []

        for s1 in critical_index:
            for s2 in final_list:
                if s1 == s2:
                    results.append(s1)

        final_list = results[:3]
        result.model("C:/Users/user/Desktop/hanlab/22_hf246/HOMEPAGE/prj/home/AccidentClassifier.h5",
                    "C:/Users/user/Desktop/hanlab/22_hf246/HOMEPAGE/prj/test.xlsx")

        return render(request, 'request/result.html', )
    else:
         return render(request, 'request/result.html', )
   

def factor(request):

    return render(request,
                  'factor/factor.html')
