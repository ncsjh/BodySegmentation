import sys
import os
import time
import random
import exposal as E
from PyQt5 import uic, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QWidget, QLabel
from PyQt5.QtCore import QThread, Qt, pyqtSignal, pyqtSlot, QByteArray, QSize
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QMovie
import torch
import cv2
import xml.etree.ElementTree as ET

import torch
import torchvision
from torchvision import models
import numpy as np

ESC_KEY=27
FRAME_RATE = 30
SLEEP_TIME = 1/FRAME_RATE
# 테스트용으로 꿀뷰를 사용해서 window_class 를 꿀뷰 클래스를 가져옴.

# camera = cv2.VideoCapture(0)
# torchvision.models.ResNet

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
    pred=[]
    doesCaptureStart=False
    def __init__(self):
        super(StartClass, self).__init__()
        self.setupUi(self)
        self.runThread = runThread(self)
        # Thread 연결
        self.runThread.changePixmap.connect(self.setImage)
        self.runThread.start()
        # 메뉴쪽 연결
        self.actionMain.triggered.connect(self.mainReturnButtonMove)
        self.actionRun.triggered.connect(self.runThread.Run)
        # self.actionStop.triggered.connect(self.stopButton)
        self.btnStart.clicked.connect(self.loading)
        self.btnStart.clicked.connect(self.startProcess)
        self.btnCapStart.clicked.connect(self.captureStart)

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

        # xml 불러오기
        xmlPath='./xml/Expose.xml'
        tree=ET.parse(xmlPath)
        root=tree.getroot()

        # expose 불러오기 / 셋
        kvp=int(expose['kvp']+2*random.randint(0,4))
        self.tedkVp.setText(str(kvp))
        ma=expose['ma']
        msec=expose['msec']*random.randrange(90, 115)/100
        mas=round(ma*msec/1000, 2)

        # xml 수정 및 저장
        root.find('kvp').text=str(kvp)
        root.find('ma').text=str(ma)
        root.find('msec').text=str(msec)
        root.find('mas').text=str(mas)
        tree.write(xmlPath)

        # expose 띄우기
        self.tedmA.setText(str(ma))
        self.tedMsec.setText(str(msec))
        self.tedmAs.setText(str(mas))
        self.tedkVp.setAlignment(Qt.AlignCenter)
        self.tedmA.setAlignment(Qt.AlignCenter)
        self.tedMsec.setAlignment(Qt.AlignCenter)
        self.tedmAs.setAlignment(Qt.AlignCenter)
        self.labelProcess.setText('분석 완료!')

        loading(self)

    def mainReturnButtonMove(self):
        widget.setCurrentIndex(widget.currentIndex()-1)

    def setMostImage(self, pixmap, isfront, prob):
        if isfront:
            if self.frontMost<prob:
                self.frontMost=prob
                self.labelFrontCapture.setPixmap(pixmap)
                self.labelFrontCapture.repaint()
                self.labelFrontProb.setText(str(prob))
                self.labelFrontProb.repaint()
        else:
            if self.sideMost<prob:
                self.sideMost=prob
                self.labelSideCapture.setPixmap(pixmap)
                self.labelSideCapture.repaint()
                self.labelSideProb.setText(str(prob))
                self.labelSideProb.repaint()


    def captureStart(self):
        self.doesCaptureStart=not self.doesCaptureStart
        if self.doesCaptureStart:
            self.btnCapStart.setText('Capt Stop')
        else:
            self.btnCapStart.setText('Capt Start')

    frontMost = 0
    sideMost = 0
    @pyqtSlot(QImage, list)
    def setImage(self, image, preds):
        self.pred=preds
        pred=0
        isFront=False
        sliderHeight=self.verticalSlider.value()
        painter1=QPainter(image)
        painter1.setPen(QPen(Qt.cyan, 10, Qt.SolidLine))
        painter1.setOpacity(0.3)
        painter1.drawLine(0, 492-sliderHeight, 229, 492-sliderHeight)
        pixmap = QPixmap.fromImage(image)
        if self.doesCaptureStart:
            if preds[0]>preds[1]:
                if preds[0]>self.frontMost:
                    isFront=True
                    pred=preds[0]
            else:
                if preds[1]>self.sideMost:
                    isFront=False
                    pred=preds[1]
            self.setMostImage(pixmap, isFront, pred)
        self.labelFrontViewVideo.setPixmap(pixmap)
        self.labelProb.setText(f'Front Prob : {preds[0]}, Side Prob : {preds[1]}')
        # lblPred=QLabel(f'Front Prob : {preds[0]}, Side Prob : {preds[1]}')

        self.labelFrontViewVideo.repaint()

class runThread(QThread):
    changePixmap = pyqtSignal(QImage, list)
    def Run(self):
        device = torch.device('cuda:0')

        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        HERE = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(HERE, 'FrontSideModel.model')

        checkpoint = torch.load(model_path)
        model = models.resnet50(num_classes=2)
        model.load_state_dict(checkpoint, strict=False)
        model.to(device)
        sideScore = 0
        frontScore = 0
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        start=time.time()
        while(True):
            ret,frame = cap.read()
            isMost=False
            isFront=False
            if ret:
                h, w, ch=frame.shape
                ws = int(w / 3)
                we = int(w * 2 / 3)

                try:
                    roi=cv2.resize(frame,(224,224))
                    rgbImage = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    h, w, ch = rgbImage.shape
                    roi = roi.astype("float") / 53

                    tf_toTensor = torchvision.transforms.ToTensor()
                    roi = tf_toTensor(roi)

                    roi = roi.unsqueeze(0)
                    roi = roi.to(device, dtype=torch.float)

                    model.eval()
                    with torch.no_grad():
                        preds = model(roi)

                        preds=preds.cpu().numpy()[0]
                        total=0
                        tr_pred=[]
                        for pred in preds:
                            total=total+np.exp(pred)
                        for pred in preds:
                            tr_pred.append(round(np.exp(pred)/total*100, 2))

                        print('\r', f'연산값 : {tr_pred}', end='')

                    rgbImage=rgbImage[:,ws:we,:]

                    # rgbImage2=rgbImage2[:,ws:we,:]
                    bytesPerLine = int(ch * w)
                    convertToQtFormat = QImage(rgbImage[0], ws,h, bytesPerLine, QImage.Format_RGB888)
                    # convertToQtFormat2 = QImage(rgbImage2[0], ws,h, bytesPerLine, QImage.Format_RGB888)

                    # bytesPerLine=ch*w
                    # convertToQtFormat=QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format_RGB888)

                    #p = convertToQtFormat.scaled(w, h, Qt.IgnoreAspectRatio)
                    # 영상 보내기

                    #todo 모델 리스트 전달 하는 곳 작업 들어가야함 . ( segmentation )
                    #처리된 결과
                    if time.time()-start>3:
                        start=time.time()

                    self.changePixmap.emit(convertToQtFormat, tr_pred)
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
    widget.setWindowTitle('X-ray 적정선량 추천AI')
    widget.setFixedWidth(1150)
    widget.setFixedHeight(680)
    widget.show()

    app.exec_()

