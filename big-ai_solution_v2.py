import os
import sys
from datetime import datetime as dt
import PySide6.QtUiTools
from PySide6.QtCore import QTimer
from PySide6.QtGui import QIcon
import pandas as pd
from PIL import Image
from detect1 import *
from detect2 import *
import shutil
from PySide6 import QtCore,QtGui
from PySide6.QtWidgets import *
from PySide6.QtGui import *
from datetime import datetime
import datetime
from datetime import date,timedelta
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
import threading
import schedule
import subprocess
from PySide6.QtGui import QPixmap

str_last = (str)
dialog_frm = None

root1 = os.getcwd()
root = root1.replace('\\','/')

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


form = resource_path("page1.ui") 
form_class = PySide6.QtUiTools.loadUiType(form)[0]

form2 = resource_path("page2.ui") 
form_class2 = PySide6.QtUiTools.loadUiType(form2)[0]

font_path = 'C:/Windows/Fonts/H2HDRM.ttf'
fontprop = fm.FontProperties(fname=font_path, size=12)



#실시간 분석 화면(page1)
class MainWindow(QMainWindow, form_class): 
    #-----------------------------------------------------------------------------------------------------------------------------------------    
    def __init__(self):
        
        super(MainWindow, self).__init__()

        self.setupUi(self)
       
        self.sheet_setup()
        flags = Qt.WindowFlags(Qt.FramelessWindowHint)  # | Qt.WindowStaysOnTopHint -> put windows on top
        self.setMaximumSize(1920, 1080)
        self.setWindowFlags(flags)
        self.setupUi(self)
        self.showMaximized()
        self.offset = None

        self.pushButton_4.clicked.connect(self.close_win)
        self.pushButton_6.clicked.connect(self.minimize_win)
        self.pushButton_5.clicked.connect(self.mini_maximize)
        
        self.df = pd.DataFrame
        self.date_check()
        
        self.count1.setText(self.count1_setup()) #금일 축관 불량 판별 갯수
        self.count3.setText(self.count3_setup()) #금일 용접 불량 판별 갯수
        
        self.count2.setText(self.count2_setup()) #전일 축관 불량 판별 갯수
        self.count4.setText(self.count4_setup()) #전일 용접 불량 판별 갯수
        
        # #매일 오전 8시에 result.csv파일에 당일날짜인덱스 생성 (ex. 230215)
        self.open_daily_csv()
        self.table_setup()
        # 고정 폴더 경로 따오기
        self.folder_root_get()
        
        # page2 페이지 여는 기능 (구현해야함!)    
        self.btn_page2.clicked.connect(self.open_window)
    
        # 폴더 내 파일 삭제하는 기능(매일 특정 시간에)
        self.timer = QTimer(self) 
        self.timer.start(1000)
        self.timer.timeout.connect(self.time_set)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.offset = event.pos()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.offset is not None and event.buttons() == QtCore.Qt.LeftButton:
            self.move(self.pos() + event.pos() - self.offset)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.offset = None
        super().mouseReleaseEvent(event)

    def close_win(self):
        self.close()

    def mini_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def minimize_win(self):
        self.showMinimized()
       
     
    #-----------------------------------------------------------------------------------------------------------------------------------------
        
    #현재 날짜 불러오기
    def today_load(self):
        now = time
        global now_str
        now_str = now.strftime('%y%m%d') #현재 시간을 str로 저장
        global now_num
        now_num = int(now_str)
        return now_num,now_str
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #어제 날짜 불러오기
    def yesterday_load(self):
        yesterday = date.today() - timedelta(1)
        if yesterday.weekday() == 5:
            yesterday = date.today() - timedelta(2) #오늘이 일요일이면 전일 판정을 금요일로 변경
        if yesterday.weekday() == 6:
            yesterday = date.today() - timedelta(3) #오늘이 월요일이면 전일 판정을 금요일로 변경
        yesterday_str = yesterday.strftime('%y%m%d')
        global yesterday_num
        yesterday_num = int(yesterday_str)
        return yesterday_num
    
    #내일 날짜 불러오기
    def tomorrow_load(self):
        tomorrow = date.today() + timedelta(1)
        tomorrow2 = date.today() + timedelta(2)
        tomorrow3 = date.today() + timedelta(3)
        tomorrow_str = tomorrow.strftime('%y%m%d')
        tomorrow2_str = tomorrow2.strftime('%y%m%d')
        tomorrow3_str = tomorrow3.strftime('%y%m%d')
        global tomorrow_num,tomorrow2_num,tomorrow3_num
        tomorrow_num = int(tomorrow_str)
        tomorrow2_num = int(tomorrow2_str)
        tomorrow3_num= int(tomorrow3_str)
        return tomorrow_num,tomorrow2_num,tomorrow3_num
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #현재 날짜 바뀌면 csv 파일에 당일 날짜 행추가하는 코드
    def date_check(self):
        running = 1
        while running:
            self.today_load()
            self.tomorrow_load()
            df1 = pd.read_csv('C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result1.csv') #용접불량
            df2 = pd.read_csv('C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result2.csv') #축관불량
            df1 = df1.astype(int)
            df2 = df2.astype(int)
            df1.set_index('date',inplace = True)
            df2.set_index('date',inplace = True)
            df2.at[tomorrow_num,'축관 불량 진단 개수'] = 0 
            df1.at[tomorrow_num,'용접 불량 진단 개수'] = 0
            df2.at[tomorrow2_num,'축관 불량 진단 개수'] = 0 
            df1.at[tomorrow2_num,'용접 불량 진단 개수'] = 0 
            df2.at[tomorrow3_num,'축관 불량 진단 개수'] = 0 
            df1.at[tomorrow3_num,'용접 불량 진단 개수'] = 0  
            df1.to_csv("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result1.csv",mode='w')
            df2.to_csv("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result2.csv",mode='w')
            if df1.loc[tomorrow_num] is not None :
                running = 0
            else:
                df2.at[now_num,'축관 불량 진단 개수'] = 0 
                df1.at[now_num,'용접 불량 진단 개수'] = 0
                df2.to_csv("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result2.csv",mode='w') 
                df1.to_csv("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result1.csv",mode='w')
                running = 0
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #로고 표시
    def sheet_setup(self):
        
        self.setWindowTitle("SD E&T Vision 플랫폼")
        self.setWindowIcon(QIcon('logo.png'))
        
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #시간 표시
    def time_set(self):
        cur_time = dt.strftime(dt.now(), "%y년 %m월 %d일 %H시 %M분")
        self.Time.setText(cur_time)
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    # 폴더 내 파일 축적 되면 삭제되는 코드
    def DeleteAllFiles(self):
        if os.path.exists(folder_root1):
            for file in os.scandir(folder_root1):
                if now_str in os.path.splitext(file)[0]:
                    continue
                else: os.remove(file.path)
        if os.path.exists(folder_root2):
            for file in os.scandir(folder_root2):
                if now_str in os.path.splitext(file)[0]:
                    continue
                else: os.remove(file.path)
                
        print ("파일 삭제 작업이 완료되었습니다. ")
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #용접 불량 실행 코드  
    def run1(self):
        
        global folder_root1
        data_path = folder_root1.replace('/','\\')
        
        global running
        running = 1
        before_len = 0
        
        # os.system('python detect1.py --source ./Store/CAM1/'+ recent_file_name + ' --exist-ok')
        
        #폴더내 파일개수 확인 및 증가 시 최근 추가된 파일 Yolo 예측 수행
        while running:
            dirList = os.listdir(data_path)
            len_dir = len(dirList)
            
            if len_dir > before_len:
                start = time.time() #running time start point
                
                before_len = len_dir
                
                file_name_lst = []
                
                
                for f_name in os.listdir(f"{data_path}"):
                    file_name_lst.append((f_name))
                    
                
                #생성시간 역순으로 정렬, 최근 생성 파일 이름 도출
                sorted_file_lst = sorted(file_name_lst, key=lambda x: x[0])
                recent_file_name = sorted_file_lst[-1]
                
                
                
                # Normal.jpg 파일만 돌아가게 만들기
                if "NORMAL" in recent_file_name:
                    
                    time.sleep(0.001)
                    #분석할 최근 이미지, 사이즈 바꿔서 빈창고에 저장
                    destination =  'C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/Store/CAM1/'+ recent_file_name #파일 복사해서 놔둘곳
                    file_path = folder_root1 + recent_file_name #파일 있는곳
                    
                    shutil.copyfile(file_path,destination)
                    time.sleep(0.001)
                    image = Image.open(destination)
                    image = image.resize((1280,1280))
                    image.save(destination)
                    
                    #yolo 분석 시작
                    # os.system('python detect1.py --source ./Store/CAM1/'+ recent_file_name + ' --exist-ok')
                    command = ["python", "detect1.py", "--source", "./Store/CAM1/" + recent_file_name, "--exist-ok"]

                    # Run detect.py as a subprocess and wait for it to finish
                    with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW) as proc:
                        for line in proc.stdout:
                            print(line, end='')  # print the output of detect.py in real-time
                            
                        # Wait for the process to finish and get the return code
                        return_code = proc.wait()
                        print(f"Process finished with return code {return_code}")
                        
                    f = open("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/warehouse1.txt", 'r')
                    
                    detect_result = f.readline()
                    f.close()
                    
                    #실시간 cam1에 detect된 결과 csv로 저장후 표출(최근 10개)시키기
                    if detect_result == 'Under':
                        new_data = {'실시간 용접 불량 진단' : '축소용접'}
                        detect_result = '축소용접'
                        self.image2_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image2_result.setStyleSheet("Color : rgb(255, 0, 0)")
                    elif detect_result == 'Over':
                        new_data = {'실시간 용접 불량 진단' : '과용접'}
                        detect_result = '과용접'
                        self.image2_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image2_result.setStyleSheet("Color : rgb(255, 0, 0)")
                    elif detect_result == 'Empty':
                        new_data = {'실시간 용접 불량 진단' : '시료없음'}
                        detect_result = '시료없음'
                        self.image2_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image2_result.setStyleSheet("Color : rgb(255, 150, 70)")
                    else : 
                        new_data = {'실시간 용접 불량 진단' : '정상'}
                        detect_result = '정상'
                        self.image2_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image2_result.setStyleSheet("Color : rgb(55, 255, 145)")
                    
                    
                    global df1
                    idx = 0 #첫번째 행에 추가
                    temp1 = df1[df1.index < idx]
                    temp2 = df1[df1.index >= idx]
                    df1 = temp1.append(new_data,ignore_index=True).append(temp2, ignore_index=True)
                    
                    if len(df1) > 10 :
                        df1.drop(df1.tail(1).index,inplace=True) 
                    
                    #detect된 이미지 불러오기
                    if detect_result is not None:
                        detect_path = 'C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/detect_store/result1/'+ recent_file_name
                        image2 = Image.open(detect_path)
                        image2 = image2.resize((600,600))
                        image2.save(detect_path)
                        self.image2.setPixmap(QPixmap(detect_path))
                        
                        self.table_setup() #실시간 10개 분석 히스토리 테이블 뜨게 하기
                        self.widget_setup()
                        
                        self.count3.setText(self.count3_setup())
                        
                        if detect_result == '과용접' :
                            pass
                        elif detect_result == '축소용접' :
                            pass
                        else:
                            self.date_check()
                            os.remove(detect_path)
                        
                        end = time.time() #running time end point
                        print("running time: "+ f"{end - start:.3f} sec") #total running time

                        
                else: 
                   before_len = len_dir
            elif len_dir <= before_len:
                before_len = len_dir
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #축관 불량 실행 코드  
    def run2(self):
        
        global folder_root2
        data_path = folder_root2.replace('/','\\')
        
        global running
        running = 1
        before_len = 0
        
        #폴더내 파일개수 확인 및 증가 시 최근 추가된 파일 Yolo 예측 수행
        while running:
            dirList = os.listdir(data_path)
            print(dirList)
            len_dir = len(dirList)
            
            if len_dir > before_len:
                start = time.time() #running time start point
                
                before_len = len_dir
                
                file_name_lst = []
                
                for f_name in os.listdir(f"{data_path}"):
                    file_name_lst.append((f_name))
                
                #생성시간 역순으로 정렬, 최근 생성 파일 이름 도출
                sorted_file_lst = sorted(file_name_lst, key=lambda x: x[0])
                recent_file_name = sorted_file_lst[-1]
                  
                #분석할 최근 이미지, 사이즈 바꿔서 빈창고에 저장
                time.sleep(0.001)
                destination = 'C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/Store/CAM2/'+ recent_file_name
                file_path = folder_root2 + recent_file_name
                shutil.copyfile(file_path,destination)
                time.sleep(0.001)
                image = Image.open(destination)
                image = image.resize((1280,1280))
                image.save(destination)
                
                #yolo 분석 시작
                # os.system('python detect2.py --source ./Store/CAM2/'+ recent_file_name + ' --exist-ok')
                command = ["python", "detect2.py", "--source", "./Store/CAM2/" + recent_file_name, "--exist-ok"]
                # Run detect.py as a subprocess and wait for it to finish
                with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW) as proc:
                    for line in proc.stdout:
                        print(line, end='')  # print the output of detect.py in real-time
                        
                    # Wait for the process to finish and get the return code
                    return_code = proc.wait()
                    print(f"Process finished with return code {return_code}")
                    
                f = open("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/warehouse2.txt", 'r')
                
                detect_result = f.readline()
                f.close()
                
                #실시간 cam2에 detect된 결과 csv로 저장후 표출(최근 10개)시키기
                
                if detect_result == 'Normal':
                    new_data = {'실시간 축관 불량 진단' : '정상'}
                    detect_result = '정상'
                    self.image1_result.setText(detect_result) #축관 불량 경과 text로 뜨게하기
                    self.image1_result.setStyleSheet("Color : rgb(55, 255, 145)")
                elif detect_result == 'Empty':
                    new_data = {'실시간 축관 불량 진단' : '시료없음'}
                    detect_result = '시료없음'
                    self.image1_result.setText(detect_result) #축관 불량 경과 text로 뜨게하기
                    self.image1_result.setStyleSheet("Color : rgb(255, 150, 70)")
                elif detect_result == 'Error':
                    new_data = {'실시간 축관 불량 진단' : '불량'}
                    detect_result = '불량'
                    self.image1_result.setText(detect_result) #축관 불량 경과 text로 뜨게하기
                    self.image1_result.setStyleSheet("Color : rgb(255, 0, 0)")
                else : 
                    new_data = {'실시간 축관 불량 진단' : '정상'}
                    detect_result = '정상'
                    self.image1_result.setText(detect_result) #축관 불량 경과 text로 뜨게하기
                    self.image1_result.setStyleSheet("Color : rgb(55, 255, 145)")
                
                global df2
                idx = 0 #첫번째 행에 추가
                temp1 = df2[df2.index < idx]
                temp2 = df2[df2.index >= idx]
                df2 = temp1.append(new_data,ignore_index=True).append(temp2, ignore_index=True)
                
                if len(df2) > 10 :
                    df2.drop(df2.tail(1).index,inplace=True)
                
                #detect된 이미지 불러오기 + daily.csv에 저장
                if detect_result is not None:
                    detect_path = 'C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/detect_store/result2/'+ recent_file_name
                    image1 = Image.open(detect_path)
                    image1 = image1.resize((600,600))
                    image1.save(detect_path)
                    self.image1.setPixmap(QPixmap(detect_path))
                    
                    self.table_setup() #실시간 10개 분석 히스토리 테이블 뜨게 하기
                    self.widget_setup()
                    
                    self.count1.setText(self.count1_setup()) #축관 불량 감지시 실시간 업데이트
                    
                    if detect_result == '불량' :
                        pass
                    else:
                        self.date_check()
                        os.remove(detect_path)
                    
                    end = time.time() #running time end point
                    print("running time: "+ f"{end - start:.3f} sec") #total running time

                else: 
                   before_len = len_dir
            elif len_dir <= before_len:
                before_len = len_dir            
                
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #고정 폴더 루트 지정하기 
    def folder_root_get(self):
        
        global folder_root1,folder_root2
        self.today_load()
        
        folder_root1 = 'D:/keyence/image/192.168.100.200/SD2/cv-x/image/SD1_001/' + now_str + '/CAM1/' # SD 경로에 맞게 수정!!!!!!!!!
        folder_root2 = 'D:/keyence/image/192.168.100.200/SD2/cv-x/image/SD1_001/' + now_str + '/CAM2/' # SD 경로에 맞게 수정!!!!!!!!!
        
        threading.Thread(target=self.run1, daemon=True).start()
        threading.Thread(target=self.run2, daemon=True).start()
        
        print(folder_root1)
        return print("실시간분석을 시작합니다.")
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #실시간으로 감지 현황 데이터프레임으로 불러오기 (최근 10개)
    def open_daily_csv(self):
        global df1,df2    
        df1 = pd.read_csv('C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/daily1.csv')
        df1.set_index('No.', inplace=True)
        df2 = pd.read_csv('C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/daily2.csv')
        df2.set_index('No.', inplace=True)
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #데이터 프레임 업데이트
    def table_setup(self):
        global df4
        df3 = pd.concat([df1,df2], axis=1)
        df4 = pd.DataFrame({'실시간 용접 불량 진단': ['-','-','-','-','-','-','-','-','-','-'],
                            '실시간 축관 불량 진단': ['-','-','-','-','-','-','-','-','-','-']})
        df4.update(df3,overwrite=True)
        return df4
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #테이블 표시
    def widget_setup(self):
        global df4
        
        self.label_11.setText(df4.iloc[0,0])
        if df4.iloc[0,0] == '시료없음' :
            self.label_11.setStyleSheet("Color : Orange;")
        elif df4.iloc[0,0] == '과용접' or df4.iloc[0,0] == '축소용접':
            self.label_11.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_11.setStyleSheet("Color : rgb(55, 255, 145)")
       
        self.label_13.setText(df4.iloc[0,1])
        if df4.iloc[0,1] == '시료없음' :
            self.label_13.setStyleSheet("Color : Orange;")
        elif df4.iloc[0,1] == '불량' :
            self.label_13.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_13.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_15.setText(df4.iloc[1,0])
        if df4.iloc[1,0] == '시료없음' :
            self.label_15.setStyleSheet("Color : Orange;")
        elif df4.iloc[1,0] == '과용접' or df4.iloc[1,0] == '축소용접':
            self.label_15.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_15.setStyleSheet("Color : rgb(55, 255, 145)")
        
        self.label_16.setText(df4.iloc[1,1])
        if df4.iloc[1,1] == '시료없음' :
            self.label_16.setStyleSheet("Color : Orange;")
        elif df4.iloc[1,1] == '불량' :
            self.label_16.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_16.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_18.setText(df4.iloc[2,0])
        if df4.iloc[2,0] == '시료없음' :
            self.label_18.setStyleSheet("Color : Orange;")
        elif df4.iloc[2,0] == '과용접' or df4.iloc[2,0] == '축소용접':
            self.label_18.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_18.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_19.setText(df4.iloc[2,1])
        if df4.iloc[2,1] == '시료없음' :
            self.label_19.setStyleSheet("Color : Orange;")
        elif df4.iloc[2,1] == '불량' :
            self.label_19.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_19.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_22.setText(df4.iloc[3,0])
        if df4.iloc[3,0] == '시료없음' :
            self.label_22.setStyleSheet("Color : Orange;")
        elif df4.iloc[3,0] == '과용접' or df4.iloc[3,0] == '축소용접':
            self.label_22.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_22.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_23.setText(df4.iloc[3,1])
        if df4.iloc[3,1] == '시료없음' :
            self.label_23.setStyleSheet("Color : Orange;")
        elif df4.iloc[3,1] == '불량' :
            self.label_23.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_23.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_27.setText(df4.iloc[4,0])
        if df4.iloc[4,0] == '시료없음' :
            self.label_27.setStyleSheet("Color : Orange;")
        elif df4.iloc[4,0] == '과용접' or df4.iloc[4,0] == '축소용접':
            self.label_27.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_27.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_28.setText(df4.iloc[4,1])
        if df4.iloc[4,1] == '시료없음' :
            self.label_28.setStyleSheet("Color : Orange;")
        elif df4.iloc[4,1] == '불량' :
            self.label_28.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_28.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_30.setText(df4.iloc[5,0])
        if df4.iloc[5,0] == '시료없음' :
            self.label_30.setStyleSheet("Color : Orange;")
        elif df4.iloc[5,0] == '과용접' or df4.iloc[5,0] == '축소용접':
            self.label_30.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_30.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_31.setText(df4.iloc[5,1])
        if df4.iloc[5,1] == '시료없음' :
            self.label_31.setStyleSheet("Color : Orange;")
        elif df4.iloc[5,1] == '불량' :
            self.label_31.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_31.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_33.setText(df4.iloc[6,0])
        if df4.iloc[6,0] == '시료없음' :
            self.label_33.setStyleSheet("Color : Orange;")
        elif df4.iloc[6,0] == '과용접' or df4.iloc[6,0] == '축소용접':
            self.label_33.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_33.setStyleSheet("Color : rgb(55, 255, 145)")
        
        self.label_34.setText(df4.iloc[6,1])
        if df4.iloc[6,1] == '시료없음' :
            self.label_34.setStyleSheet("Color : Orange;")
        elif df4.iloc[6,1] == '불량' :
            self.label_34.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_34.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_36.setText(df4.iloc[7,0])
        if df4.iloc[7,0] == '시료없음' :
            self.label_36.setStyleSheet("Color : Orange;")
        elif df4.iloc[7,0] == '과용접' or df4.iloc[7,0] == '축소용접':
            self.label_36.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_36.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_37.setText(df4.iloc[7,1])
        if df4.iloc[7,1] == '시료없음' :
            self.label_37.setStyleSheet("Color : Orange;")
        elif df4.iloc[7,1] == '불량' :
            self.label_37.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_37.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_39.setText(df4.iloc[8,0])
        if df4.iloc[8,0] == '시료없음' :
            self.label_39.setStyleSheet("Color : Orange;")
        elif df4.iloc[8,0] == '과용접' or df4.iloc[8,0] == '축소용접':
            self.label_39.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_39.setStyleSheet("Color : rgb(55, 255, 145)")
            
        self.label_40.setText(df4.iloc[8,1])
        if df4.iloc[8,1] == '시료없음' :
            self.label_40.setStyleSheet("Color : Orange;")
        elif df4.iloc[8,1] == '불량' :
            self.label_40.setStyleSheet("Color : rgb(255, 0, 0)")
        else :
            self.label_40.setStyleSheet("Color : rgb(55, 255, 145)")

        return df4
    
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #당일,전일 불량 판별 개수 저장한 csv 파일 불러오기
    def count_setup1(self):
        global count_df1
        count_df1 = pd.read_csv('C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result1.csv')
        count_df1.set_index('date',inplace=True)
        count_df1.astype('int')
        return count_df1
    
    def count_setup2(self):
        global count_df2
        count_df2 = pd.read_csv('C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/result2.csv')
        count_df2.set_index('date',inplace=True)
        count_df2.astype('int')
        return count_df2
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #count1(금일 축관 불량 판별 개수) 불러오기
    def count1_setup(self):
        self.count_setup2()
        self.today_load()
        return str(count_df2.loc[now_num,'축관 불량 진단 개수'])
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #count2(전일 축관 불량 판별 개수) 불러오기
    def count2_setup(self):
        self.count_setup2()
        self.yesterday_load()
        return str(count_df2.loc[yesterday_num,'축관 불량 진단 개수'])
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #count3(금일 용접 불량 판별 개수) 불러오기
    def count3_setup(self):
        self.count_setup1()
        self.today_load()
        return str(count_df1.loc[now_num,'용접 불량 진단 개수'])
    #-----------------------------------------------------------------------------------------------------------------------------------------
    #count4(전일 용접 불량 판별 개수) 불러오기
    def count4_setup(self):
        self.count_setup1()
        self.yesterday_load()
        return str(count_df1.loc[yesterday_num,'용접 불량 진단 개수'])
    
    #-------------------------------------------------------------------------------------------------------------------------------------
    def open_window(self):
        second_window = Secondwindow()
        second_window.show()
        second_window.exec_()  
            
        
#-----------------------------------------------------------------------------------------------------------------------------------------
class Secondwindow(QDialog,form_class2):
    def __init__(self):
        super().__init__()
        
        self.setupUi(self)

        self.sheet_setup()

        self.df = pd.DataFrame

        self.btn_new.clicked.connect(self.data_get) #분석 시작 파일 버튼
        
        self.timer = QTimer(self) 
        self.timer.start(1000)
        self.timer.timeout.connect(self.time_set)
       
        self.btn_open_file.clicked.connect(self.open_data) #분석된 파일이 담긴 폴더 오픈
        self.default_directory = 'C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/Store/CAM1' #불량으로 담긴 폴더로 고정경로 설정
        
        self.btn_page1.clicked.connect(self.close)
        
           
    def sheet_setup(self):    
        self.setWindowTitle("SD E&T Vision 플랫폼")
        self.setWindowIcon(QIcon('logo.png'))

    def time_set(self):
        cur_time = dt.strftime(dt.now(), "%y년 %m월 %d일 %H시 %M분")
        self.Time.setText(cur_time)

    def data_get(self):
        file_dialog = QFileDialog()
        file_dialog.setDirectory(self.default_directory)
        dialog_frm = QFileDialog.getOpenFileName(self)
        try:
            if dialog_frm[0]:
                start = time.time() #running time start point
                
                file_path=dialog_frm[0]
                file_path_str = file_path.split('/')
                str_last = file_path_str[-1]
                destination = root + '/user_want_detect/'+ str_last
                shutil.copyfile(file_path,destination)
                image = Image.open(destination)
                image = image.resize((1280,1280))
                image.save(destination)
                
                #Cam1 detect time text
                name_box = str_last.split('_')
                global cam1_detect_time
                cam1_detect_time = name_box[0] + name_box[1]
                
                if os.path.isfile(destination):
                    command = ["python", "detect1.py", "--source", "./user_want_detect/" + str_last,"--project", "./user_detect_store","--name","result","--exist-ok"]
                    # Run detect.py as a subprocess and wait for it to finish
                    subprocess.run(command,creationflags=subprocess.CREATE_NO_WINDOW)
                    # os.system('python detect1.py --source ./user_want_detect/'+ str_last + ' --project ./user_detect_store --name result --exist-ok')
                    #detect.py 분석후 detect된 이미지 user_detect_store 안에 result 하위 폴더(고정)로 저장
                     
                    f = open("C:/Users/202210/Desktop/Python Project/yolov5_sdent_detect_model/warehouse1.txt", 'r')
                    

                    detect_result = f.readline()
                    f.close()
                    if detect_result == 'Under':
                        detect_result = '축소용접'
                        self.image3_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image3_result.setStyleSheet("Color : rgb(255, 0, 0)")
                    elif detect_result == 'Over':
                        detect_result = '과용접'
                        self.image3_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image3_result.setStyleSheet("Color : rgb(255, 0, 0)")
                    elif detect_result == 'Empty':
                        detect_result = '시료없음'
                        self.image3_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image3_result.setStyleSheet("Color : rgb(255, 150, 70)")
                    else : 
                        detect_result = '정상'
                        self.image3_result.setText(detect_result) #용접 불량 경과 text로 뜨게하기
                        self.image3_result.setStyleSheet("Color : rgb(55, 255, 145)")
                    
                    if detect_result is not None:
                        global detect_path
                        detect_path = root +'/user_detect_store/result/'+ str_last
                        image4 = Image.open(detect_path)
                        image4 = image4.resize((580,580))
                        image4.save(detect_path)
                        self.image4.setPixmap(QPixmap(detect_path))
                        
                        #Cam2 image load
                        file_date = str_last.split('_')
                        file_date_month = file_date[0]
                        
                        #file_change
                        def listToString(str_list):
                            result = ""
                            for s in str_list:
                                result += s
                            return result.strip()
                        
                        #Cam1 detect image file text
                        box = listToString(cam1_detect_time)
                        cam1_detect_time = box[0:2]+'년 '+box[2:4]+'월 '+box[4:6]+'일 '+box[6:8]+'시 '+box[8:10]+'분 '+box[10:12]+'초 '
                        print(cam1_detect_time)
                        
                        
                        str_list = file_date_month
                        change_str_filedate = listToString(str_list)
                        # dir = 'D:/keyence/image/192.168.100.200/SD2/cv-x/image/SD1_001/CAM2'  <------ 이 코드로 수정 필요! ## SD E&T 고정 경로
                        #-----------------------------------------------------------------------------------------------------------------------------
                        dir = 'D:/keyence/image/192.168.100.200/SD2/cv-x/image/SD1_001/'+ change_str_filedate + '/CAM2' #나중에는 없애야함 (실험용 코드) # SD 경로에 맞게 수정!!!!!!!!!
                        #-----------------------------------------------------------------------------------------------------------------------------
                        files = os.listdir(dir)
                        
                        #Cam1 image datetime load
                        cam1_image_hour = file_date[1]
                        str_list = cam1_image_hour
                        
                        cam1_image_hour = listToString(str_list)
                        cam1_image_hour_format = '%H%M%S'
                        cam1_datetime = datetime.datetime.strptime(cam1_image_hour,cam1_image_hour_format)
                        
                        cam1_datetime_0 = cam1_datetime - datetime.timedelta(seconds=26)
                        cam1_datetime_1 = cam1_datetime - datetime.timedelta(seconds=27)
                        cam1_datetime_2 = cam1_datetime - datetime.timedelta(seconds=28)
                        cam1_datetime_3 = cam1_datetime - datetime.timedelta(seconds=29)
                        cam1_datetime_4 = cam1_datetime - datetime.timedelta(seconds=30)
                        cam1_datetime_5 = cam1_datetime - datetime.timedelta(seconds=31)
                        cam1_datetime_6 = cam1_datetime - datetime.timedelta(seconds=32)
                        cam1_datetime_7 = cam1_datetime - datetime.timedelta(seconds=33)
                        cam1_datetime_8 = cam1_datetime - datetime.timedelta(seconds=34)
                        cam1_datetime_9 = cam1_datetime - datetime.timedelta(seconds=35)
                        cam1_datetime_10 = cam1_datetime - datetime.timedelta(seconds=36)
                        cam1_datetime_11 = cam1_datetime - datetime.timedelta(seconds=37)
                        cam1_datetime_12 = cam1_datetime - datetime.timedelta(seconds=38)
                        cam1_datetime_13 = cam1_datetime - datetime.timedelta(seconds=39)
                        cam1_datetime_14 = cam1_datetime - datetime.timedelta(seconds=40)
                        
                        #Cam2 image = Cam1 image, Check!
                        global file_name_result
                        file_name_result = (str)
                        for file in files:
                            file_name = os.path.basename(file)
                            file_name_total = os.path.basename(file)
                            file_name = file_name.split('_')
                            file_hour = file_name[1]
                            str_list = file_hour
                            file_hour = listToString(str_list)
                            file_hour_format = '%H%M%S'
                            datetime_result = datetime.datetime.strptime(file_hour, file_hour_format)
                            if ( datetime_result == cam1_datetime_0):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_1):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_2):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_3):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_4):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_5):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_6):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_7):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_8):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_9):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_10):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_11):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_12):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_13):
                                file_name_result = file_name_total
                                break
                            elif (datetime_result == cam1_datetime_14):
                                file_name_result = file_name_total
                                break
                        
                        if file_name_result != (str) : 
                            cam2_image_root = dir + '/' + file_name_result
                            
                            file_name_result = file_name_result.split('_')
                            
                            box1 = file_name_result[0]+file_name_result[1]
                            
                            cam2_detect_time = box1[0:2]+'년 '+box1[2:4]+'월 '+box1[4:6]+'일 '+box1[6:8]+'시 '+box1[8:10]+'분 '+box1[10:12]+'초 '
                            
                            self.image3_time.setText(cam2_detect_time) 
                            self.image4_time.setText(cam1_detect_time) 
                            
                            image3 = Image.open(cam2_image_root)
                            image3 = image3.resize((580,580))
                            image3.save(cam2_image_root)
                            self.image3.setPixmap(QPixmap(cam2_image_root))
                            
                            #data
                            img_date = datetime.datetime.strptime(box1, '%y%m%d%H%M%S%f')
                            img_date_middle = img_date.strftime("%Y.%m.%d %H:%M:%S:000")
                            
                            #-------------------------------------------------------------------------------------------------------------------------------------------------------------
                            data = pd.read_csv('D:/Trend/'+'SCADA' + img_date.strftime("%Y-%m-%d") + '.csv', sep='\t',encoding='UTF-16') #csv 파일이 담긴 폴더 결로(수정필요)
                            data = data[['TIME','PHASE_01.TMP.REAL','PHASE_01.PRE.REAL','AI.TEMP_RAW','PHASE_02.TMP.REAL','PHASE_02.PRE.REAL']]
                            #-------------------------------------------------------------------------------------------------------------------------------------------------------------
                            if data is None :
                                self.graph1.setText("이미지 없음")
                                self.graph2.setText("이미지 없음")
                                self.graph3.setText("이미지 없음")
                                self.image3.setText("이미지 없음")
                                self.image2_result.setText("해당하는 CSV 파일 없음")
                                self.image2_result.setStyleSheet("Color : Yellow;")
                                self.image2_result.setFont(QtGui.QFont("에스코어 드림 6 Bold", 38))
                                print("분석하고 싶은 이미지를 다시 선택하여 주시길 바랍니다.")
                                
                            index_number = data.index[(data['TIME']==img_date_middle)]
                            index_number_value = index_number[0]
                            
                            global graph_start_index,graph_end_index
                            graph_start_index = None
                            graph_end_index = None
                            
                            if index_number_value is not None:
                                if index_number_value <= 600:
                                    graph_start_index = 0
                                    graph_end_index = index_number_value + 600
                                else :
                                    graph_start_index = index_number_value - 600
                                    graph_end_index = index_number_value + 600
                            
                            data = data.iloc[graph_start_index:graph_end_index,1:]
                            data.astype(float)
                            
                            len_data = 1200
                            plus1 = 0
                            plus2 = 0
                            plus3 = 0 #최종 데이터 분석 결과값
                            
                            for i in range(len_data):
                                if i == 0:
                                    i += 1
                                    continue
                                
                                pre_data1 = data.loc[graph_start_index+i-1,'PHASE_01.PRE.REAL']
                                cur_data1 = data.loc[graph_start_index+i,'PHASE_01.PRE.REAL']
                                if cur_data1 < pre_data1:
                                    plus1 = plus1+1
                                else :
                                    plus1 = 0
                                
                                pre_data2 = data.loc[graph_start_index+i-1,'PHASE_02.PRE.REAL']
                                cur_data2 = data.loc[graph_start_index+i,'PHASE_02.PRE.REAL']
                                if cur_data2 < pre_data2:
                                    plus2 = plus2+1
                                else :
                                    plus2 = 0
                                
                                if plus1 > 30 :
                                    plus3 += 1
                                    plus1 = 0
                                if plus2 > 30 :
                                    plus3 += 1
                                    plus2 = 0
                                    
                                i += 1
                                
                            if plus3 >= 1:
                                result = "공정 정지이력 존재"
                                self.image2_result.setText(result)
                                self.image2_result.setStyleSheet("Color : rgb(255, 0, 0)")
                                self.image2_result.setFont(QtGui.QFont("에스코어 드림 6 Bold", 50)) 
                            else:
                                result = "정상 운전"
                                self.image2_result.setText(result)
                                self.image2_result.setStyleSheet("Color : rgb(55, 255, 145)")
                                self.image2_result.setFont(QtGui.QFont("에스코어 드림 6 Bold", 50))     
                            
                        
                            fig1 = plt.figure(figsize=(6, 4), dpi=200)
                            fig2 = plt.figure(figsize=(6, 4), dpi=200)
                            fig3 = plt.figure(figsize=(6, 4), dpi=200)
                            ax1 = fig1.add_subplot(111)
                            ax2 = fig2.add_subplot(111)
                            ax3 = fig3.add_subplot(111)

                            # 전체 배경색 설정
                            color = (0, 0, 20/255)
                            fig1.patch.set_facecolor(color)
                            fig2.patch.set_facecolor(color)
                            fig3.patch.set_facecolor(color)
                            ax1.set_facecolor(color)
                            ax2.set_facecolor(color)
                            ax3.set_facecolor(color)

                            # 그래프 그리기
                            ax1.plot(data['PHASE_01.TMP.REAL'], label='PHASE_01.TMP.REAL')
                            ax1.plot(data['PHASE_02.TMP.REAL'], 'orange', label='PHASE_02.TMP.REAL')
                            ax1.set_xlabel("")
                            ax1.set_ylabel("Temperature", color='white',fontsize=16)
                            ax1.legend(labels=['1차 축관 온도', '2차 축관 온도'], loc='upper right',prop=fontprop)
                            ax1.spines['bottom'].set_color('white') # x축 색을 흰색으로 설정합니다.
                            ax1.spines['left'].set_color('white') # y축 색을 흰색으로 설정합니다.
                            ax1.tick_params(axis='x', colors='white',labelsize=13) # x축 눈금 색을 흰색으로 설정합니다.
                            ax1.tick_params(axis='y', colors='white',labelsize=13) # y축 눈금 색을 흰색으로 설정합니다.
                            ax1.yaxis.label.set_color('white') # y축 레이블 색을 흰색으로 설정합니다.
                            ax1.xaxis.label.set_color('white') # x축 레이블 색을 흰색으로 설정합니다.
                            ax1.title.set_color('white') # 제목 색을 흰색으로 설정합니다.
                            fig1.savefig('raw_plot_1.jpg', dpi=200)
                            
                            ax2.plot(data['PHASE_01.PRE.REAL'], label='PHASE_01.PRE.REAL')
                            ax2.plot(data['PHASE_02.PRE.REAL'], 'orange', label='PHASE_02.PRE.REAL')
                            ax2.set_xlabel("")
                            ax2.set_ylabel("Pressure", color='white',fontsize=16)
                            ax2.legend(labels=['1차 축관 압력', '2차 축관 압력'], loc='upper right',prop=fontprop)
                            ax2.spines['bottom'].set_color('white') # x축 색을 흰색으로 설정합니다.
                            ax2.spines['left'].set_color('white') # y축 색을 흰색으로 설정합니다.
                            ax2.tick_params(axis='x', colors='white',labelsize=13) # x축 눈금 색을 흰색으로 설정합니다.
                            ax2.tick_params(axis='y', colors='white',labelsize=13) # y축 눈금 색을 흰색으로 설정합니다.
                            ax2.yaxis.label.set_color('white') # y축 레이블 색을 흰색으로 설정합니다.
                            ax2.xaxis.label.set_color('white') # x축 레이블 색을 흰색으로 설정합니다.
                            ax2.title.set_color('white') # 제목 색을 흰색으로 설정합니다.
                            fig2.savefig('raw_plot_2.jpg', dpi=200)
                            
                            ax3.plot(data['AI.TEMP_RAW'], label='AI.TEMP_RAW')
                            ax3.set_xlabel("")
                            ax3.set_ylabel("Temperature", color='white',fontsize=16)
                            ax3.legend(labels=['1차 축관부 온도'], loc='upper right',prop=fontprop)
                            ax3.spines['bottom'].set_color('white') # x축 색을 흰색으로 설정합니다.
                            ax3.spines['left'].set_color('white') # y축 색을 흰색으로 설정합니다.
                            ax3.tick_params(axis='x', colors='white',labelsize=13) # x축 눈금 색을 흰색으로 설정합니다.
                            ax3.tick_params(axis='y', colors='white',labelsize=13) # y축 눈금 색을 흰색으로 설정합니다.
                            ax3.yaxis.label.set_color('white') # y축 레이블 색을 흰색으로 설정합니다.
                            ax3.xaxis.label.set_color('white') # x축 레이블 색을 흰색으로 설정합니다.
                            ax3.title.set_color('white') # 제목 색을 흰색으로 설정합니다.
                            fig3.savefig('raw_plot_3.jpg', dpi=200)


                            self.qPixmapFileVar_1 = QPixmap()
                            self.qPixmapFileVar_1.load("raw_plot_1.jpg")
                            self.qPixmapFileVar_2 = QPixmap()
                            self.qPixmapFileVar_2.load("raw_plot_2.jpg")
                            self.qPixmapFileVar_3 = QPixmap()
                            self.qPixmapFileVar_3.load("raw_plot_3.jpg")

                            self.graph1.setPixmap(self.qPixmapFileVar_1)
                            self.graph1.setScaledContents(True)
                            self.graph2.setPixmap(self.qPixmapFileVar_2)
                            self.graph2.setScaledContents(True)
                            self.graph3.setPixmap(self.qPixmapFileVar_3)
                            self.graph3.setScaledContents(True)
                        else :
                            self.graph1.setText("이미지 없음")
                            self.graph2.setText("이미지 없음")
                            self.graph3.setText("이미지 없음")
                            self.image3.setText("이미지 없음")
                            self.image2_result.setText("해당하는 축관 이미지 없음")
                            self.image2_result.setStyleSheet("Color : Yellow;")
                            self.image2_result.setFont(QtGui.QFont("에스코어 드림 6 Bold", 38))
                            print("분석하고 싶은 이미지를 다시 선택하여 주시길 바랍니다.")
                            
                        end = time.time() #running time end point
                        print("running time: "+ f"{end - start:.3f} sec") #total running time

                return file_path
            

        except Exception as e:
            print(e)
        

    def open_data(self):     
        root1 = os.getcwd()     
        os.startfile(root1 +'\\user_detect_store\\')
        
  
if __name__ == "__main__":
    app = QApplication(sys.argv)
    mw1 = MainWindow()
    mw1.showMaximized()
    
    def run_schedule():
        schedule.run_pending()
        QTimer.singleShot(1000, run_schedule)
    QTimer.singleShot(1000, run_schedule)
    
    sys.exit(app.exec())
    
    # 처음으로 Git Hub에 올려보았습니다.