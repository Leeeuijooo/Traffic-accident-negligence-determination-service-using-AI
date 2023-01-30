from pathlib import Path
from unittest import result
import pandas as pd
import re

import csv
import pandas as pd

#########################
###### df 생성 ##########


########################

def check(a_list,df):
  critical_factor = []
  if '0' in a_list:
    df['초록불'] = 1
    critical_factor.append('초록불')

  if '1' in a_list:
    df['좌회전신호'] = 1
    critical_factor.append('좌회전신호')
    
  if '2' in a_list:
    df['좌측깜박이'] = 1
    critical_factor.append('좌측깜박이')

  if '3' in a_list:
    df['속도100'] = 1
    critical_factor.append('속도100')

  if '4' in a_list:
    df['속도110'] = 1
    critical_factor.append('속도110')

  if '5' in a_list:
    df['속도50'] = 1
    critical_factor.append('속도50')

  if '6' in a_list:
    df['빨간불'] = 1
    critical_factor.append('빨간불')

  if '7' in a_list:
    df['우측깜박이'] = 1
    critical_factor.append('우측깜박이')

  if '8' in a_list:
    df['유턴'] = 1
    critical_factor.append('유턴')

  if '9' in a_list:
    df['노란불'] = 1
    critical_factor.append('노란불')

  if '10' in a_list:
    df['중앙분리대'] = 1
    critical_factor.append('중앙분리대')

  if '11' in a_list:
    df['회전교차로'] = 1
    critical_factor.append('회전교차로')

  # 12는 횡단보도 : 스키마 파일에 X

  if '13' in a_list:
    df['감소도로 화살표'] = 1
    critical_factor.append('감소도로 화살표')    

  if '14' in a_list:
    df['합류도로 노면표시'] = 1
    critical_factor.append('류도로 노면표시')

  if '15' in a_list:
    df['역삼각형'] = 1
    critical_factor.append('역삼각형')
    
  if '16' in a_list:
    df['유도봉'] = 1
    critical_factor.append('유도봉')
    
  if '17' in a_list:
    df['동시신호'] = 1
    critical_factor.append('동시신호')
    
  if '18' in a_list:
    df['직진 후 좌회전'] = 1
    critical_factor.append('직진 후 좌회전')
        
  if '100' in a_list:
    df['내가 좌회전'] = 1
    critical_factor.append('내가 좌회전')
        
  if '200' in a_list:
    df['내가 우회전'] = 1
    critical_factor.append('내가 우회전')
    
  if '300' in a_list:
    df['내차 차선변경'] = 1
    critical_factor.append('내차 차선변경')
        
  if '400' in a_list:
    df['상대방차 차선 변경'] = 1
    critical_factor.append('상대방차 차선 변경')
    
  # 나머지 0으로 채워주기
  df = df.fillna(0)

  return df, critical_factor

def make(a_list):
    
    df = pd.DataFrame(index=range(0,1), columns = ['내차 차선변경', '상대방차 차선 변경','내가 우회전','내가 좌회전','합류도로 노면표시',
                   '유도봉','좌측깜박이','우측깜박이','빨간불','노란불','초록불','좌회전신호',
                   '감소도로 화살표','속도100','속도110','속도50','동시신호','직진 후 좌회전','유턴',
                   '중앙분리대','회전교차로','역삼각형','과실비율','사고 번호'])

    df = df[['내차 차선변경', '상대방차 차선 변경', '내가 우회전', '내가 좌회전', '합류도로 노면표시',
             '유도봉', '좌측깜박이', '우측깜박이', '빨간불', '노란불', '초록불', '좌회전신호',
                    '감소도로 화살표', '속도100', '속도110', '속도50', '동시신호', '직진 후 좌회전', '유턴',
                    '중앙분리대', '회전교차로', '역삼각형', '과실비율', '사고 번호']]

    ########################
    ########### test #######

    df, critical_factor = check(a_list, df)

    ########################

    df.to_excel("test.xlsx",  encoding="utf-8-sig")
    return(critical_factor)