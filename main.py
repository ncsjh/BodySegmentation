import sys
import os
import time
import random
import exposal as E
from PyQt5 import uic,QtWidgets
from PyQt5.QtWidgets import  QApplication,QMainWindow,QDialog
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
from PyQt5.QtGui import QImage, QPixmap

import exposal as E
import cv2

#path 구분자
def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

## 메인 페이지 연결
mainform = resource_path('UI/form_Main.ui')
mainFormClass = uic.loadUiType(mainform)[0]
## 시작 페이지 연결
startform = resource_path('UI/form_Start.ui')
startFormClass = uic.loadUiType(startform)[0]
## 셋팅 페이지 연결
isFinished=False
#메인 페이지 전체 클래스 = UI 연동
class WindowClass(QMainWindow,mainFormClass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #연결
        self.StartButton.clicked.connect(self.startButtonMove)

    def startButtonMove(self):
        widget.setCurrentIndex(widget.currentIndex()+1)

#스타트 페이지 전체 클래스 = UI 연동
class StartClass(QMainWindow,startFormClass):
    def __init__(self):
        super(StartClass, self).__init__()
        self.setupUi(self)
        self.runThread = runThread(self);
        # Thread 연결
        self.runThread.changePixmap.connect(self.setImage)
        self.runThread.start()
        # 메뉴쪽 연결
        self.actionMain.triggered.connect(self.mainReturnButtonMove)
        self.actionRun.triggered.connect(self.runThread.Run)
        # self.actionStop.triggered.connect(self.stopButton)
        # self.btnStart.clicked.connect(self.runThread.run)
        self.syncComboBox()
        self.tedkVp.setAlignment(Qt.AlignCenter)
        self.tedmA.setAlignment(Qt.AlignCenter)
        self.tedMsec.setAlignment(Qt.AlignCenter)
        self.tedmAs.setAlignment(Qt.AlignCenter)
        views=E.getViews()
        for view in views:
            self.cbView.addItem(view)
        self.cbView.setCurrentText('CHEST')
        positions=E.getPositions(self.cbView.currentText())
        for position in positions:
            self.cbPosition.addItem(position)
        # self.cbView.get
        self.cbView.currentIndexChanged.connect(self.syncComboBox)


    # def stopButton(self):
    #     self.runThread.stop()
    #     self.runThread.wait(1000)
    #     print("중지 되었습니다.")
    def syncComboBox(self) :
        self.cbPosition.clear()
        positions=E.getPositions(self.cbView.currentText())
        for position in positions:
            self.cbPosition.addItem(position)

    def mainReturnButtonMove(self):
        widget.setCurrentIndex(widget.currentIndex()-1)

    @pyqtSlot(QImage, QImage, bool)
    def setImage(self, image,image2, isFinished):
        pixmap = QPixmap.fromImage(image)
        self.labelSideViewVideo.setPixmap(pixmap)
        self.labelSideViewVideo.repaint()
        pixmap2 = QPixmap.fromImage(image2)
        self.labelFrontViewVideo.setPixmap(pixmap2)
        self.labelFrontViewVideo.repaint()
        if isFinished:
            expose = E.getExposal(self.cbView.currentText(), self.cbPosition.currentText())
            self.tedkVp.setText(str(expose['kvp']))
            self.tedmA.setText(str(expose['ma']))
            self.tedMsec.setText(str(expose['msec']))
            self.tedmAs.setText(str(expose['mas']))
            self.tedkVp.setAlignment(Qt.AlignCenter)
            self.tedmA.setAlignment(Qt.AlignCenter)
            self.tedMsec.setAlignment(Qt.AlignCenter)
            self.tedmAs.setAlignment(Qt.AlignCenter)

class runThread(QThread):
    changePixmap = pyqtSignal(QImage, QImage,bool)

    def Run(self):
        isFinished=False
        #print("영상 촬영 시작")
        cap = cv2.VideoCapture(0)
        cap2 = cv2.VideoCapture(1)
        start=time.time()
        while(True):
            ret,frame = cap.read()
            ret2,frame2=cap2.read()
            if time.time() - start<3:
                isFinished=False
            if ret:
                h, w, ch=frame.shape
                ws = int(w / 3)
                we = int(w * 2 / 3)

                rgbImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                rgbImage2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
                h, w, ch = rgbImage.shape

                rgbImage=rgbImage[:,ws:we,:]
                rgbImage2=rgbImage2[:,ws:we,:]
                bytesPerLine = int(ch * w)
                convertToQtFormat = QImage(rgbImage[0], ws,h, bytesPerLine, QImage.Format_RGB888)
                convertToQtFormat2 = QImage(rgbImage2[0], ws,h, bytesPerLine, QImage.Format_RGB888)

                # bytesPerLine=ch*w
                # convertToQtFormat=QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)

                #p = convertToQtFormat.scaled(w, h, Qt.IgnoreAspectRatio)
                # 영상 보내기

                #todo 모델 리스트 전달 하는 곳 작업 들어가야함 . ( sgrmantation )

                #처리된 결과
                if time.time()-start>3:
                    isFinished=True
                    start=time.time()

                #결과 값 전달.
                #영상  , 관전압 결과 , 간전류 결과
                self.changePixmap.emit(convertToQtFormat,convertToQtFormat2, isFinished)
                # 임의 결과값 출력

                cv2.waitKey(0)

    def randomResult(self):
        return random.randrange(80,120)

if __name__ == "__main__":
    #QApplication : 프로그램 실행 시키는 클래스
    app = QApplication(sys.argv)

    #화면 전환용 Widget 설정
    widget = QtWidgets.QStackedWidget()

    #레이아웃 인스턴스 생성
    mainWindow = WindowClass()
    StartWindow = StartClass()
    #위젯 추가 ( 연결 )
    widget.addWidget(mainWindow)
    widget.addWidget(StartWindow)
    # 프로그램 화면 보이는 코드
    # 위제 사이즈 지정
    widget.setWindowTitle('전신영상 기반 체형 예측 후 최적의 X-ray 조건을 추천AI')
    widget.setFixedWidth(854)
    widget.setFixedHeight(629)
    widget.show()

    app.exec_()

