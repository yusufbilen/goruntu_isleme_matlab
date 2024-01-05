from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QGraphicsScene, QGraphicsPixmapItem, QProgressBar
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import QTimer
import time
import detection
from openFile import *
# 1-7 satırları arasında verilen kodlarda yolov5 için gerrekli kütüphaneler içe aktarılmıştır

class MyClass(QMainWindow):  
    # MyClass adında bir sınıf tanımlanıyor
    def __init__(self):  
        # Oluşturucu metod, "init"  ile başlamalı
        super().__init__()  
        # Üst sınıfın (QMainWindow) oluşturucu metodunu çağırıyor

        # Ui_Dialog sınıfından nesne oluşturma (örnek olarak burada adını "ui" olarak varsayalım)
        self.ui = Ui_MainWindow()
        # 'Ui_MainWindow' sınıfından bir örnek oluşturuluyor ve 'self.ui' değişkenine atanıyor.
        self.ui.setupUi(self)
        # Oluşturulan arayüz 'setupUi()' yöntemi kullanılarak belirtilen pencereye (self) ayarlanıyor.        self.ui.pushButton.clicked.connect(self.getImage)
        # Bir PyQt5 uygulamasında, QPushButton nesneleri tıklama olaylarına sahiptir.
        # pushButton adlı düğmeye tıklandığında getImage() adlı bir işlevi bağlar.
        self.ui.pushButton_2.clicked.connect(self.detectionPath)
        # Benzer şekilde, pushButton_2 adlı diğer bir düğmeye tıklandığında detectionPath() işlevini bağlar.
        self.ui.pushButton_3.clicked.connect(self.clearPath)
        # pushButton_3 adlı bir başka düğmeye tıklandığında clearPath() işlevini bağlar.
        self.pbar = QProgressBar(self)
        # Bir QProgressBar örneği oluşturuluyor ve bu öğe, belirtilen pencereye (self) ekleniyor.
        self.pbar.setGeometry(QtCore.QRect(220, 685, 561, 31))
        # ProgressBar'un konumu ve boyutu ayarlanıyor (sol üst köşe x, y, genişlik, yükseklik olarak).
        self.pbar.setStyleSheet("background-color: rgb(81, 81, 81);")
        # ProgressBar'un arka plan rengi stil özelliği üzerinden griye ayarlanıyor.
        self.ui.pushButton_2.setEnabled(False)
        # ui içerisindeki 'pushButton_2' adlı buton devre dışı bırakılıyor.
        # Pencerenin boyutunu sabit yapın
        self.setFixedSize(1515, 770)
          # İstediğiniz sabit boyutu ayarlayabilirsiniz

        
    def clearPath(self):
        self.ui.graphicsView.setScene(None)
        # graphicsView'e bağlı sahne (scene) temizleniyor, yani grafik görüntü boşaltılıyor.
        self.ui.graphicsView_2.setScene(None)
        # graphicsView_2'ye bağlı sahne (scene) temizleniyor, yani grafik görüntü boşaltılıyor.
        self.updateProgressBar(0)
        # Bir ilerleme çubuğu (ProgressBar) güncelleme işlevi çağrılıyor ve 0 değeri ile güncelleniyor.
    def getImagePath(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', 'c:', "Image files (*.jpg *.gif)")
        # Dosya açma iletişim kutusunu çağırır ve varsayılan olarak 'c:' dizinindeki (*.jpg *.gif) uzantılı resim dosyalarını açar.
        return fname[0]
        # Dosya adının yer aldığı bir demet (tuple) döndürülür ve buradan sadece dosya adı alınarak döndürülüyor.

    def getImage(self):
        self.getImagePath()
        # getImagePath() fonksiyonunu çağırarak kullanıcıdan resim dosyasının yolunu alır.

        if self.imagePath:
        # imagePath değerinin varlığını kontrol ederek, bir resmin yüklenip yüklenmediğini kontrol eder.
            QtWidgets.QGraphicsScene(self)
        # Yeni bir QGraphicsScene örneği oluşturur. QGraphicsScene, grafik nesnelerinin sahne üzerinde düzenlenmesini sağlar.
            QPixmap(self.imagePath)
            # Seçilen resmin yolu kullanılarak QPixmap sınıfından bir örnek oluşturulur. QPixmap, resimleri yükleme, gösterme ve düzenleme işlevselliği sağlar.

        
            # Resmi istediğiniz genişlik ve yükseklikte ayarlayın
            new_width = 701  # Yeni genişlik
            new_height = 571  # Yeni yükseklik
            pixmap = pixmap.scaled(new_width, new_height)
            # Yüklenen pixmap'i, belirtilen yeni genişlik ve yükseklikte ölçekler.
            item = QtWidgets.QGraphicsPixmapItem(pixmap)
            # QGraphicsPixmapItem sınıfı, QGraphicsScene içinde bir resmi göstermek için kullanılır. Bir QPixmap nesnesi alır ve bu nesneyi görsel olarak temsil eder.
            scene.addItem(item)
            # Oluşturulan QGraphicsPixmapItem'i sahneye ekler.
            self.ui.graphicsView.setScene(scene)
            # Sahneyi oluşturulan sahneyle ilişkilendirir, bu da resmi görüntülemek için kullanılacak olan QGraphicsView bileşenine bağlanır.
            self.ui.graphicsView.setStyleSheet("border: none;")
            # QGraphicsView bileşeninin kenarlığını kaldırmak için bir CSS stil tanımlar.
            self.ui.pushButton_2.setEnabled(True)
            # İlgili pushButton'un etkinleştirilmesini sağlar. 
    def detectionImage(self):
        self.updateProgressBar(0)
        # detectionImage fonksiyonu, bir nesne algılama işlemi başlatılmadan önce ilerleme çubuğunu sıfırlar.
        detection.run(source=self.imagePath, exist_ok=True)
        # Nesne algılama işlemini başlatır
        for i in range(101):
            # İlerleme çubuğunu güncelleer
            time.sleep(0.05)
            # bekleme süresini ayarlar
            self.updateProgressBar(i)
            # ilerleme çubuğunu günceller
        
    def detectionPath(self):
        self.image_Path = self.detectionImage()
        # 'detectionImage' fonksiyonunu çağırıp dönen değeri 'self.image_Path' değişkenine atar
        newPath = self.imagePath.replace("data/images", "runs/exp")
        # 'imagePath' değişkenindeki "data/images" stringini "runs/exp" ile değiştirerek yeni bir yol oluşturur
        if self.imagePath:
            scene = QtWidgets.QGraphicsScene(self)
            # QGraphicsScene sınıfından yeni bir sahne oluşturur
            pixmap = QPixmap(newPath)
            # Yeni oluşturulan 'newPath' yoluyla QPixmap'ten bir pixmap oluşturur
            # Resmi istediğiniz genişlik ve yükseklikte ayarlayın
            new_width = 701  # Yeni genişlik
            new_height = 571  # Yeni yükseklik
            pixmap = pixmap.scaled(new_width, new_height)
            # Yüklenen pixmap'i, belirtilen yeni genişlik ve yükseklikte ölçekler.
            scene.addItem(item)
            # Oluşturulan QGraphicsPixmapItem'i sahneye ekler.
            self.ui.graphicsView.setScene(scene)
            # Sahneyi oluşturulan sahneyle ilişkilendirir, bu da resmi görüntülemek için kullanılacak olan QGraphicsView bileşenine bağlanır.
            self.ui.graphicsView.setStyleSheet("border: none;")
            # QGraphicsView bileşeninin kenarlığını kaldırmak için bir CSS stil tanımlar.
            self.ui.pushButton_2.setEnabled(True)
            # İlgili pushButton'un etkinleştirilmesini sağlar.

    def updateProgressBar(self, value):
        self.pbar.setValue(value)
         # Bu metod, 'pbar' adlı ilerleme çubuğunun değerini 'value' ile günceller
# Bu kod parçası bir Qt uygulamasının ana döngüsünü başlatmak için kullanılır
app = QtWidgets.QApplication([])
# 'QtWidgets.QApplication([])' ile bir Qt uygulaması oluşturulur
dialog = MyClass()
# PyQt5 uygulamasının ana döngüsünü başlatır.
# 'MyClass()' sınıfından bir nesne oluşturulur
dialog.show()
# Dialog penceresini görünür kılar
app.exec_()
# Uygulamanın ana döngüsünü başlatır
