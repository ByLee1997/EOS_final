#!/usr/bin/env python
import sys
import socket
#from PySide2.QtWidgets import QApplication, QMainWindow
#from PySide2 import QtWidgets, QtCore, QtGui

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.setupUi(self)


    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1127, 605)
        self.centralWidget = QtWidgets.QWidget(MainWindow)
        self.centralWidget.setObjectName("centralWidget")
        self.dog_name_1 = QtWidgets.QLabel(self.centralWidget)
        self.dog_name_1.setGeometry(QtCore.QRect(200, 100, 67, 17))
        self.dog_name_1.setObjectName("dog_name_1")
        self.dog_place_1 = QtWidgets.QTextBrowser(self.centralWidget)
        self.dog_place_1.setGeometry(QtCore.QRect(110, 330, 256, 31))
        self.dog_place_1.setObjectName("dog_place_1")
        self.dog_today_eat_time = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_eat_time.setGeometry(QtCore.QRect(110, 410, 81, 21))
        self.dog_today_eat_time.setObjectName("dog_today_eat_time")
        self.label_1_3 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_3.setGeometry(QtCore.QRect(200, 410, 51, 21))
        self.label_1_3.setObjectName("label_1_3")
        self.dog_avg_eat_time_1 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_eat_time_1.setGeometry(QtCore.QRect(253, 410, 71, 23))
        self.dog_avg_eat_time_1.setObjectName("dog_avg_eat_time_1")
        self.label_1_4 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_4.setGeometry(QtCore.QRect(330, 410, 41, 21))
        self.label_1_4.setObjectName("label_1_4")
        self.dog_avg_drink_time_1 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_drink_time_1.setGeometry(QtCore.QRect(253, 440, 71, 23))
        self.dog_avg_drink_time_1.setObjectName("dog_avg_drink_time_1")
        self.label_1_6 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_6.setGeometry(QtCore.QRect(330, 440, 41, 21))
        self.label_1_6.setObjectName("label_1_6")
        self.label_1_5 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_5.setGeometry(QtCore.QRect(200, 440, 51, 21))
        self.label_1_5.setObjectName("label_1_5")
        self.dog_today_drink_time_1 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_drink_time_1.setGeometry(QtCore.QRect(110, 440, 81, 21))
        self.dog_today_drink_time_1.setObjectName("dog_today_drink_time_1")
        self.label_1_1 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_1.setGeometry(QtCore.QRect(110, 380, 67, 17))
        self.label_1_1.setObjectName("label_1_1")
        self.label_1_2 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_2.setGeometry(QtCore.QRect(250, 380, 67, 17))
        self.label_1_2.setObjectName("label_1_2")
        self.label_eat = QtWidgets.QLabel(self.centralWidget)
        self.label_eat.setGeometry(QtCore.QRect(40, 410, 67, 17))
        self.label_eat.setObjectName("label_eat")
        self.label_drink = QtWidgets.QLabel(self.centralWidget)
        self.label_drink.setGeometry(QtCore.QRect(40, 440, 67, 17))
        self.label_drink.setObjectName("label_drink")
        self.label_1_7 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_7.setGeometry(QtCore.QRect(527, 410, 51, 21))
        self.label_1_7.setObjectName("label_1_7")
        self.dog_place_2 = QtWidgets.QTextBrowser(self.centralWidget)
        self.dog_place_2.setGeometry(QtCore.QRect(437, 330, 256, 31))
        self.dog_place_2.setObjectName("dog_place_2")
        self.label_1_8 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_8.setGeometry(QtCore.QRect(657, 440, 41, 21))
        self.label_1_8.setObjectName("label_1_8")
        self.label_1_9 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_9.setGeometry(QtCore.QRect(577, 380, 67, 17))
        self.label_1_9.setObjectName("label_1_9")
        self.dog_name_2 = QtWidgets.QLabel(self.centralWidget)
        self.dog_name_2.setGeometry(QtCore.QRect(527, 100, 67, 17))
        self.dog_name_2.setObjectName("dog_name_2")
        self.dog_today_eat_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_eat_time_2.setGeometry(QtCore.QRect(437, 410, 81, 21))
        self.dog_today_eat_time_2.setObjectName("dog_today_eat_time_2")
        self.dog_avg_drink_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_drink_time_2.setGeometry(QtCore.QRect(580, 440, 71, 23))
        self.dog_avg_drink_time_2.setObjectName("dog_avg_drink_time_2")
        self.label_1_10 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_10.setGeometry(QtCore.QRect(437, 380, 67, 17))
        self.label_1_10.setObjectName("label_1_10")
        self.dog_avg_eat_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_eat_time_2.setGeometry(QtCore.QRect(580, 410, 71, 23))
        self.dog_avg_eat_time_2.setObjectName("dog_avg_eat_time_2")
        self.label_1_11 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_11.setGeometry(QtCore.QRect(527, 440, 51, 21))
        self.label_1_11.setObjectName("label_1_11")
        self.dog_today_drink_time_2 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_drink_time_2.setGeometry(QtCore.QRect(437, 440, 81, 21))
        self.dog_today_drink_time_2.setObjectName("dog_today_drink_time_2")
        self.label_1_12 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_12.setGeometry(QtCore.QRect(657, 410, 41, 21))
        self.label_1_12.setObjectName("label_1_12")
        self.label_1_13 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_13.setGeometry(QtCore.QRect(860, 410, 51, 21))
        self.label_1_13.setObjectName("label_1_13")
        self.dog_place_3 = QtWidgets.QTextBrowser(self.centralWidget)
        self.dog_place_3.setGeometry(QtCore.QRect(770, 330, 256, 31))
        self.dog_place_3.setObjectName("dog_place_3")
        self.label_1_14 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_14.setGeometry(QtCore.QRect(990, 440, 41, 21))
        self.label_1_14.setObjectName("label_1_14")
        self.label_1_15 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_15.setGeometry(QtCore.QRect(910, 380, 67, 17))
        self.label_1_15.setObjectName("label_1_15")
        self.dog_name_3 = QtWidgets.QLabel(self.centralWidget)
        self.dog_name_3.setGeometry(QtCore.QRect(860, 100, 67, 17))
        self.dog_name_3.setObjectName("dog_name_3")
        self.dog_today_eat_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_eat_time_3.setGeometry(QtCore.QRect(770, 410, 81, 21))
        self.dog_today_eat_time_3.setObjectName("dog_today_eat_time_3")
        self.dog_avg_drink_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_drink_time_3.setGeometry(QtCore.QRect(913, 440, 71, 23))
        self.dog_avg_drink_time_3.setObjectName("dog_avg_drink_time_3")
        self.label_1_16 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_16.setGeometry(QtCore.QRect(770, 380, 67, 17))
        self.label_1_16.setObjectName("label_1_16")
        self.dog_avg_eat_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_avg_eat_time_3.setGeometry(QtCore.QRect(913, 410, 71, 23))
        self.dog_avg_eat_time_3.setObjectName("dog_avg_eat_time_3")
        self.label_1_17 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_17.setGeometry(QtCore.QRect(860, 440, 51, 21))
        self.label_1_17.setObjectName("label_1_17")
        self.dog_today_drink_time_3 = QtWidgets.QLCDNumber(self.centralWidget)
        self.dog_today_drink_time_3.setGeometry(QtCore.QRect(770, 440, 81, 21))
        self.dog_today_drink_time_3.setObjectName("dog_today_drink_time_3")
        self.label_1_18 = QtWidgets.QLabel(self.centralWidget)
        self.label_1_18.setGeometry(QtCore.QRect(990, 410, 41, 21))
        self.label_1_18.setObjectName("label_1_18")
        self.dog_fig_1 = QtWidgets.QLabel(self.centralWidget)
        self.dog_fig_1.setGeometry(QtCore.QRect(110, 130, 251, 171))
        self.dog_fig_1.setText("")
        self.dog_fig_1.setPixmap(QtGui.QPixmap("dog1.jpg"))
        self.dog_fig_1.setScaledContents(True)
        self.dog_fig_1.setWordWrap(False)
        self.dog_fig_1.setObjectName("dog_fig_1")
        self.dog_fig_2 = QtWidgets.QLabel(self.centralWidget)
        self.dog_fig_2.setGeometry(QtCore.QRect(440, 130, 251, 171))
        self.dog_fig_2.setText("")
        self.dog_fig_2.setPixmap(QtGui.QPixmap("dog2.jpg"))
        self.dog_fig_2.setScaledContents(True)
        self.dog_fig_2.setObjectName("dog_fig_2")
        self.dog_fig_3 = QtWidgets.QLabel(self.centralWidget)
        self.dog_fig_3.setGeometry(QtCore.QRect(770, 130, 251, 171))
        self.dog_fig_3.setText("")
        self.dog_fig_3.setPixmap(QtGui.QPixmap("dog3.jpg"))
        self.dog_fig_3.setScaledContents(True)
        self.dog_fig_3.setObjectName("dog_fig_3")
        MainWindow.setCentralWidget(self.centralWidget)
        self.menuBar = QtWidgets.QMenuBar(MainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 1127, 28))
        self.menuBar.setObjectName("menuBar")
        self.menuDOG = QtWidgets.QMenu(self.menuBar)
        self.menuDOG.setObjectName("menuDOG")
        MainWindow.setMenuBar(self.menuBar)
        self.mainToolBar = QtWidgets.QToolBar(MainWindow)
        self.mainToolBar.setObjectName("mainToolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.mainToolBar)
        self.statusBar = QtWidgets.QStatusBar(MainWindow)
        self.statusBar.setObjectName("statusBar")
        MainWindow.setStatusBar(self.statusBar)
        self.menuBar.addAction(self.menuDOG.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "DOG"))
        self.dog_name_1.setText(_translate("MainWindow", "Dog1"))
        self.label_1_3.setText(_translate("MainWindow", "times  /"))
        self.label_1_4.setText(_translate("MainWindow", "times"))
        self.label_1_6.setText(_translate("MainWindow", "times"))
        self.label_1_5.setText(_translate("MainWindow", "times /"))
        self.label_1_1.setText(_translate("MainWindow", "Today"))
        self.label_1_2.setText(_translate("MainWindow", "Average"))
        self.label_eat.setText(_translate("MainWindow", "Eat"))
        self.label_drink.setText(_translate("MainWindow", "Drink"))
        self.label_1_7.setText(_translate("MainWindow", "times  /"))
        self.label_1_8.setText(_translate("MainWindow", "times"))
        self.label_1_9.setText(_translate("MainWindow", "Average"))
        self.dog_name_2.setText(_translate("MainWindow", "Dog2"))
        self.label_1_10.setText(_translate("MainWindow", "Today"))
        self.label_1_11.setText(_translate("MainWindow", "times /"))
        self.label_1_12.setText(_translate("MainWindow", "times"))
        self.label_1_13.setText(_translate("MainWindow", "times  /"))
        self.label_1_14.setText(_translate("MainWindow", "times"))
        self.label_1_15.setText(_translate("MainWindow", "Average"))
        self.dog_name_3.setText(_translate("MainWindow", "Dog3"))
        self.label_1_16.setText(_translate("MainWindow", "Today"))
        self.label_1_17.setText(_translate("MainWindow", "times /"))
        self.label_1_18.setText(_translate("MainWindow", "times"))
        self.menuDOG.setTitle(_translate("MainWindow", "DOG"))



if __name__ == "__main__":
    app = QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
