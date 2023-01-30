import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openpyxl
import csv

from sklearn.linear_model import LogisticRegression  # logistic regression
from sklearn import svm  # support vector Machine
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.naive_bayes import GaussianNB  # Naive bayes
from sklearn.tree import DecisionTreeClassifier  # Decision Tree
# training and testing data split
from sklearn.model_selection import train_test_split
from sklearn import metrics  # accuracy measure
from sklearn.metrics import confusion_matrix  # for confusion matrix

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils


def model(model_pile, excel_pile):
    model = tf.keras.models.load_model(model_pile)

    # 엑셀 파일 불러오기
    df = pd.read_excel(excel_pile, index_col=0)

    # 열 이름 변경
    df.rename(columns={'내차 차선변경': 'changeLane', '상대방차 차선 변경': 'theothercar_ChangeLane', '내가 우회전': 'rightTurn', '내가 좌회전': 'leftTurn', '합류도로 노면표시': 'mergeroad',
                       '유도봉': 'safetyrod', '좌측깜박이': 'Leftblinker', '우측깜박이': 'Rightblinker', '빨간불': 'REDlight', '노란불': 'YELLOWlight', '초록불': 'GREENlight', '좌회전신호': 'LEFTlight',
                       '감소도로 화살표': 'decreasedRoad', '속도100': 'Limit100', '속도110': 'Limit110', '속도50': 'Limit50', '동시신호': 'simultaneousSignal', '직진 후 좌회전': 'straightLeft', '유턴': 'Uturn',
                       '중앙분리대': 'median', '회전교차로': 'circle', '역삼각형': 'retriangle', '과실비율': 'percentage', '사고 번호': 'Accident'}, inplace=True)

    # Y값 설정 (사고번호)
    Y = df['Accident']
    Class = ['201(A)', '201(B)', '262(A)', '262(B)', '252(A)', '252(B)',
             '252-2', '252-3(B)', '252-3(A)', '501(A)', '501(B)', '507(A)',
             '507(B)']
    # Class = Y.unique()

    # 딕셔너리 생성 (%와 사고번호 값 연결)
    percent = {'201(A)': 0, '201(B)': 100, '262(A)': 80, '262(B)': 20, '252(A)': 30, '252(B)': 70, '252-2': 50, '252-3(B)': 0,
               '252-3(A)': 100, '501(A)': 30, '501(B)': 70, '507(A)': 0, '507(B)': 100}

    # 열 제거
    # df=df.drop(['파일명','비고','rightTurn','leftTurn','percentage'],axis=1)
    df = df.drop(['rightTurn', 'leftTurn', 'percentage', 'Accident'], axis=1)
    #df = df.drop(['Unnamed: 0'], axis = 1, inplace = True)
    # print(df)
    #X = df.drop('Accident', axis=1)

    print(df.columns)

    pred = model.predict(df, batch_size=32, verbose=12)

    '''
  p = pred[0]

  for i, acc in enumerate(p) :
      print(Class[i], "=", int(acc*100), "%")
  print("---")
  print("예측한 결과 = " , Class[p.argmax()])
  '''

    for i in range(0, 1):
        result_list = []
        p = pred[i]
        print(i, "번째 영상 예측 결과 및 과실 비율")
        print('')

        result_list.append(i)
        for j, acc in enumerate(p):
            print(Class[j], "=", int(acc*100), "%")

        result = Class[p.argmax()]
        result_percent = percent[result]

        result_list.append(result)
        result_list.append(result_percent)

        print("----------")
        print("예측한 결과 = ", Class[p.argmax()])
        print("예측 과실 비율 = ", result_percent)
        print('')
        print("결과 리스트 : ", result_list)
        print("----------")
        print('')

    with open("result.csv", 'w') as file:
        writer = csv.writer(file)
        writer.writerow(result_list)
