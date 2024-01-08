# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources: # verilerin nereden aldığını belirtmektedir
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam  # bu komut detect.py dosyasında ki nesne tanımlama modülünün çalışmasını sağlar.
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

# bu kdolamalar çalışacak olan yolov5 modülünün çalıştırdığı ve kapsadığı dosya veya bağlamtıların modellerini temsil ediyor.

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""
# Bu komutla, YOLOv5 modelinin farklı formatlarda dönüştürülmüş versiyonlarını kullanarak nesne ayrıntılarını gerçekleştirmeye olanak tanır

import argparse
import os
import platform
import sys
from pathlib import Path

import torch
# 31-35 gerekli olan kütüphaneleri içe aaktarır

FILE = Path(__file__).resolve()  # Bu satır, mevcut dosyanın yolunu alır ve çözümler.
ROOT = FILE.parents[0]  # YOLOv5 kök dizini. Dosyanın üst dizinine erişir.
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # ROOT'u PATH'e ekler.
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # Kökü mevcut çalışma dizinine göre göreceli hale getirir.

# YOLOv5 kodu içeriği
from ultralytics.utils.plotting import Annotator, colors, save_one_box  # Ultralytics kütüphanesinden gerekli modülleri alır.

from models.common import DetectMultiBackend  # Ortak modellerden 'DetectMultiBackend' modülünü alır.
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams  # Veri yükleyici modülleri.
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
# Genel yardımcı işlevlerin olduğu modülü içeriye dahil eder.
from utils.torch_utils import select_device, smart_inference_mode
# PyTorch ile ilgili yardımcı işlevleri içeriye dahil eder.



@smart_inference_mode() #bir sınıf içindeki bir metodu veya işlevi dekore etmek için kullanılır
def run( #run adında bir sınıf tanımlanır
    weights=ROOT / 'yolov5s.pt',  # YOLOv5 modelinin ağırlıklarının dosya yolu veya triton URL
    source=ROOT / 'data/images',  # Algılama yapılacak olan dosya, dizin, URL, glob, ekran görüntüsü veya web kamerası kaynağı
    data=ROOT / 'data/coco128.yaml',  # Veri seti yapılandırma dosyasının (dataset.yaml) yolu
    imgsz=(640, 640),  # Çıkarım boyutları (yükseklik, genişlik) - görüntü boyutu
    conf_thres=0.25,  # Güven eşiği (confidence threshold)
    iou_thres=0.45,  # NMS (Non-Maximum Suppression) IOU eşiği
    max_det=1000,  # Her bir görüntüdeki maksimum tespit sayısı
    device='',  # CUDA cihazı, örneğin 0 veya 0,1,2,3 veya cpu
    view_img=False,  # Sonuçları görüntüleme
    save_txt=False,  # *.txt olarak sonuçları kaydetme
    save_conf=False,  # --save-txt etiketlerinde güvenleri kaydetme
    save_crop=False,  # Kesilmiş tahmin kutularını kaydetme
    nosave=False,  # Görüntüleri/videoları kaydetmeme
    classes=None,  # Sınıfa göre filtreleme: --class 0 veya --class 0 2 3
    agnostic_nms=False,  # Sınıf-agnostic NMS
    augment=False,  # Artırılmış çıkarım
    visualize=False,  # Özellikleri görselleştirme
    update=False,  # Tüm modelleri güncelleme
    project=ROOT / 'runs/detect',  # Sonuçları projeye/adı kaydetme
    name='exp',  # Sonuçları proje/ad ile kaydetme
    exist_ok=False,  # Var olan proje/ad geçerli, arttırmama
    line_thickness=3,  # Sınırlayıcı kutu kalınlığı (piksel)
    hide_labels=False,  # Etiketleri gizleme
    hide_conf=False,  # Güvenleri gizleme
    half=False,  # FP16 yarı-kesinlikli çıkarım kullanma
    dnn=False,  # ONNX çıkarımı için OpenCV DNN kullanma
    vid_stride=1,  # Video kare hızı adımı
):
    source = str(source)  # Kaynağı bir dizeye dönüştürür.
    save_img = not nosave and not source.endswith('.txt')  # Çıkarım görüntülerini kaydetme koşulu
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)  # Dosya uzantısını kontrol eder
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))  # Kaynağın URL olup olmadığını kontrol eder
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)  # Web kamerasından veya akışlardan görüntü alınıp alınmadığını kontrol eder
    screenshot = source.lower().startswith('screen')  # Ekran görüntüsü alınıp alınmadığını kontrol eder

    if is_url and is_file:
        source = check_file(source)  # İndirme işlemi

    # Dizinler
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # Çalışmayı artırır
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # Dizin oluşturur

    # Modeli Yükleme
    device = select_device(device)  # Cihazı seçer
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)  # Modeli yükler
    stride, names, pt = model.stride, model.names, model.pt  # Adımlama, isimler ve pt (model) bilgilerini alır
    imgsz = check_img_size(imgsz, s=stride)  # Görüntü boyutunu kontrol eder

    # Veri Yükleyici
    bs = 1  # Toplu iş boyutu
    if webcam:
        view_img = check_imshow(warn=True)  # Görüntüyü gösterme koşulunu kontrol eder
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # Akışları yükler
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)  # Ekran görüntülerini yükler
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)  # Görüntüleri yükler
    vid_path, vid_writer = [None] * bs, [None] * bs  # Video yolu ve yazıcılarını belirler


    # Çıkarım yapma
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # Modeli hazırlama

    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())  # Değişkenlerin başlangıç değerleri

    for path, im, im0s, vid_cap, s in dataset:  # Veri kümesindeki her bir veri için döngü
        with dt[0]:  # Zaman ölçümü için
            im = torch.from_numpy(im).to(model.device)  # Görüntüyü tensora dönüştürme ve cihaza yükleme
            im = im.half() if model.fp16 else im.float()  # Veri tipini ayarlama (fp16 veya float32)
            im /= 255  # 0 - 255 aralığındaki piksel değerlerini 0.0 - 1.0 aralığına normalizasyon
            if len(im.shape) == 3:
                im = im[None]  # Tek bir görüntü için boyutu genişletme (batch dim)

        # Çıkarım işlemi
        with dt[1]:  # Zaman ölçümü için
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False  # Görüntüleri kaydetme yolu
            pred = model(im, augment=augment, visualize=visualize)  # Model üzerinden çıkarım yapma

        # NMS (Non-Maximum Suppression)
        with dt[2]:  # Zaman ölçümü için
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)  # Çıktıları filtreleme (NMS)

        # Tahminleri işleme
        for i, det in enumerate(pred):  # Her bir görüntü için işlem yapma
            seen += 1  # Görüntü sayısını artırma
            if webcam:  # web kamerasından geliyorsa (batch_size >= 1)
                p, im0, frame = path[i], im0s[i].copy(), dataset.count  # Yolu, görüntüyü ve kare sayısını alır
                s += f'{i}: '  # Sayaç güncelleme
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  # Yolu, görüntüyü ve kare sayısını alır

        p = Path(p)  # Yolu 'Path' nesnesine dönüştürme
        save_path = str(save_dir / p.name)  # Kayıt edilecek görüntü yolunu oluşturma (im.jpg)
        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # Etiket dosyasının yolu (im.txt)
        s += '%gx%g ' % im.shape[2:]  # Yazdırılacak metni oluşturma (görüntü boyutları)

        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalizasyon faktörü (whwh)
        imc = im0.copy() if save_crop else im0  # save_crop için görüntü kopyası oluşturma

        annotator = Annotator(im0, line_width=line_thickness, example=str(names))  # Etiketleme için annotator oluşturma

        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()  # Kutuları img_size boyutundan im0 boyutuna yeniden ölçeklendirme

            for c in det[:, 5].unique():  # Her sınıf için tekrar sayısını ve sınıf adını metne ekler
                n = (det[:, 5] == c).sum()  # Sınıf bazında algılama sayısı
                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # Metne ekleme

            for *xyxy, conf, cls in reversed(det):  # Her bir algılama için işlem yapma
                if save_txt:  # Dosyaya yazma
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # Normalized xywh
                    line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # Etiket formatı
                    with open(f'{txt_path}.txt', 'a') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')  # Etiketleri dosyaya yazma

                if save_img or save_crop or view_img:  # bbox'ı görüntüye ekleme
                    c = int(cls)  # Sınıfın integer değeri
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')  # Etiket
                    annotator.box_label(xyxy, label, color=colors(c, True))  # bbox'ı görüntüye ekleme
                    if save_crop:  # bbox'ı kaydetme
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

        # Sonuçları görüntüye ekleme ve görüntüleme
        im0 = annotator.result()
        if view_img:  # Görüntüyü gösterme
            if platform.system() == 'Linux' and p not in windows:
                windows.append(p)
                cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # Pencere boyutunu ayarlama (Linux)
                cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            cv2.imshow(str(p), im0)
            cv2.waitKey(1)  # 1 milisaniyelik bekleme süresi

        # Sonuçları kaydetme (algılamalı görüntü)
        if save_img:  
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)  # Görüntüyü kaydetme
            else:  # 'video' veya 'stream'
                if vid_path[i] != save_path:  # Yeni bir video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # Önceki video yazarını serbest bırakma
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # Sonuç videosuna *.mp4 uzantısı verme
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))  # Video yazarı oluşturma
                vid_writer[i].write(im0)  # Videoya görüntüyü yazma

        # Zamanı yazdırma (yalnızca çıkarım süresi)
        LOGGER.info(f"{s}{'' if len(det) else '(tespit yok), '}{dt[1].dt * 1E3:.1f}ms")

        # Sonuçları yazdırma
        t = tuple(x.t / seen * 1E3 for x in dt)  # her bir görüntü için hızlar
        LOGGER.info(f'Speed: %.1fms ön işleme, %.1fms çıkarım, %.1fms NMS, her bir görüntüde (boyut: {(1, 3, *imgsz)})' % t)

        # Sonuçları kaydetme
        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} etiket {save_dir / 'labels'} klasörüne kaydedildi" if save_txt else ''
            LOGGER.info(f"Sonuçlar {colorstr('bold', save_dir)} klasörüne kaydedildi{s}")

        # Model güncellemesi
        if update:
            strip_optimizer(weights[0])  # modeli güncelle (SourceChangeWarning'u düzeltmek için)


def parse_opt():
    # Argümanları analiz etmek için bir argüman analizcisi oluşturma
    parser = argparse.ArgumentParser()

    # Argümanları tanımlama
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()  # Argümanları ayrıştırma
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # imgsz boyutunu genişletme
    print_args(vars(opt))  # Argümanları yazdırma
    return opt  # Argümanları döndürme


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))  # Gerekli kütüphaneleri kontrol etme
    run(**vars(opt))  # Çıkarım işlemini çalıştırma


if __name__ == '__main__':
    opt = parse_opt()  # Argümanları ayrıştırma
    main(opt)  # Ana işlevi çağırma
