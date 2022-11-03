import sys
import os
import time
import random
import exposal as E
from PyQt5 import uic,QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QByteArray, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QMovie

import exposal as E
import cv2

#path 구분자
def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)
loadform=resource_path('UI/load.ui')
FROM_CLASS_Loading = uic.loadUiType(loadform)[0]
## 메인 페이지 연결
mainform = resource_path('UI/form_Main.ui')
mainFormClass = uic.loadUiType(mainform)[0]
## 시작 페이지 연결
startform = resource_path('UI/form_Start.ui')
startFormClass = uic.loadUiType(startform)[0]
## 셋팅 페이지 연결
isFinished=False
#메인 페이지 전체 클래스 = UI 연동
sliderHeight=0
class WindowClass(QMainWindow,mainFormClass):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #연결
        self.StartButton.clicked.connect(self.startButtonMove)

    def startButtonMove(self):
        widget.setCurrentIndex(widget.currentIndex()+1)


class loading(QWidget, FROM_CLASS_Loading):

    def __init__(self, parent):
        super(loading, self).__init__(parent)
        self.setupUi(self)
        self.center()
        self.cntPreset=150
        self.show()

        # 동적 이미지 추가
        self.movie = QMovie('pngegg.png', QByteArray(), self)
        self.movie.setScaledSize(QSize(200,200))
        self.movie.setCacheMode(QMovie.CacheAll)
        # QLabel에 동적 이미지 삽입
        self.label.setMovie(self.movie)
        self.movie.start()
        # 윈도우 해더 숨기기
        self.setWindowFlags(Qt.FramelessWindowHint)

    # 위젯 정중앙 위치
    def center(self):
        size = self.size()
        ph = self.parent().geometry().height()
        pw = self.parent().geometry().width()
        self.move(int(pw / 2 - size.width() / 2), int(ph / 2 - size.height() / 2))

    def showEvent(self, event):
        self.timer = self.startTimer(30)

        self.counter = 0
    def timerEvent(self, event):
        self.counter += 1
        self.update()
        self.label.setText(str(self.cntPreset - self.counter))
        if self.counter == self.cntPreset:  # 300번 호출되면
            self.killTimer(self.timer)  # 타이머 종료하고
        self.hide()

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
        self.btnStart.clicked.connect(self.loading)

        self.btnStart.clicked.connect(self.startProcess)



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

    def syncComboBox(self) :
        self.cbPosition.clear()
        positions=E.getPositions(self.cbView.currentText())
        for position in positions:
            self.cbPosition.addItem(position)

    def loading(self):
        try:
            self.loading
            self.loading.deleteLater()
        # 처음 클릭하는 경우
        except:
            self.loading = loading(self)
    def startProcess(self):
        start=time.time()

        expose = E.getExposal(self.cbView.currentText(), self.cbPosition.currentText())
        self.labelProcess.setText('분석중.')
        self.labelProcess.repaint()
        time.sleep(0.5)
        self.labelProcess.setText('분석중..')
        self.labelProcess.repaint()
        time.sleep(0.5)
        self.labelProcess.setText('분석중...')
        self.labelProcess.repaint()
        time.sleep(0.5)
        self.tedkVp.setText(str(expose['kvp']*random.randrange(95, 105)/100))
        ma=expose['ma']*random.randrange(95, 105)/100
        msec=expose['msec']*random.randrange(95, 105)/100
        self.tedmA.setText(str(ma))
        self.tedMsec.setText(str(msec))
        self.tedmAs.setText(str(round(ma*msec/1000, 2)))
        self.tedkVp.setAlignment(Qt.AlignCenter)
        self.tedmA.setAlignment(Qt.AlignCenter)
        self.tedMsec.setAlignment(Qt.AlignCenter)
        self.tedmAs.setAlignment(Qt.AlignCenter)
        self.labelProcess.setText('분석 완료!')
        loading(self)


    def mainReturnButtonMove(self):
        widget.setCurrentIndex(widget.currentIndex()-1)

    @pyqtSlot(QImage, QImage, bool)
    def setImage(self, image,image2, isFinished):
        sliderHeight=self.verticalSlider.value()
        painter1=QPainter(image)
        painter2=QPainter(image2)
        painter1.setPen(QPen(Qt.cyan, 10, Qt.SolidLine))
        painter2.setPen(QPen(Qt.cyan, 10, Qt.SolidLine))
        painter2.setOpacity(0.3)
        painter1.setOpacity(0.3)
        painter1.drawLine(0, 492-sliderHeight, 229, 492-sliderHeight)
        painter2.drawLine(0, 492-sliderHeight, 229, 492-sliderHeight)
        pixmap = QPixmap.fromImage(image)
        pixmap2 = QPixmap.fromImage(image2)
        self.labelSideViewVideo.setPixmap(pixmap)
        self.labelSideViewVideo.repaint()
        self.labelFrontViewVideo.setPixmap(pixmap2)
        self.labelFrontViewVideo.repaint()

class runThread(QThread):
    changePixmap = pyqtSignal(QImage, QImage,bool)
    def Run(self):
        isFinished=False
        print("영상 촬영 시작")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        print('카메라 1 불러오기')
        cap2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
        print('카메라 2 불러오기')
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

                try:
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

                    #todo 모델 리스트 전달 하는 곳 작업 들어가야함 . ( segmentation )

                    #처리된 결과
                    if time.time()-start>3:
                        isFinished=True
                        start=time.time()

                    #결과 값 전달.
                    #영상  , 관전압 결과 , 간전류 결과
                    self.changePixmap.emit(convertToQtFormat,convertToQtFormat2, isFinished)
                # 임의 결과값 출력
                except:
                    pass


                cv2.waitKey(0)
        cap.release()
        print('릴리즈 됨')
        cap2.release()


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
    widget.setFixedHeight(680)
    widget.show()

    app.exec_()

