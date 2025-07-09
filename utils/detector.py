# utils/detector.py
import os
import cv2
from ultralytics import YOLO

# Load model YOLOv12/YOLOv8 (pre-trained)
model = YOLO("yolo12x.pt")  # pastikan model sudah didownload

#region 📦 Daftar Kelas COCO (80 class) - Untuk model YOLO bawaan

# 0: person  
# 1: bicycle  
# 2: car  
# 3: motorcycle  
# 4: airplane  
# 5: bus  
# 6: train  
# 7: truck  
# 8: boat  
# 9: traffic light  
# 10: fire hydrant  
# 11: stop sign  
# 12: parking meter  
# 13: bench  
# 14: bird  
# 15: cat  
# 16: dog  
# 17: horse  
# 18: sheep  
# 19: cow  
# 20: elephant  
# 21: bear  
# 22: zebra  
# 23: giraffe  
# 24: backpack  
# 25: umbrella  
# 26: handbag  
# 27: tie  
# 28: suitcase  
# 29: frisbee  
# 30: skis  
# 31: snowboard  
# 32: sports ball  
# 33: kite  
# 34: baseball bat  
# 35: baseball glove  
# 36: skateboard  
# 37: surfboard  
# 38: tennis racket  
# 39: bottle  
# 40: wine glass  
# 41: cup  
# 42: fork  
# 43: knife  
# 44: spoon  
# 45: bowl  
# 46: banana  
# 47: apple  
# 48: sandwich  
# 49: orange  
# 50: broccoli  
# 51: carrot  
# 52: hot dog  
# 53: pizza  
# 54: donut  
# 55: cake  
# 56: chair  
# 57: couch  
# 58: potted plant  
# 59: bed  
# 60: dining table  
# 61: toilet  
# 62: tv  
# 63: laptop  
# 64: mouse  
# 65: remote  
# 66: keyboard  
# 67: cell phone  
# 68: microwave  
# 69: oven  
# 70: toaster  
# 71: sink  
# 72: refrigerator  
# 73: book  
# 74: clock  
# 75: vase  
# 76: scissors  
# 77: teddy bear  
# 78: hair drier  
# 79: toothbrush

#endregion

#region 🤖 Macam-macam YOLO dan Kegunaannya

# YOLOv5: Populer dan stabil. Cocok untuk proyek real-time biasa.
# YOLOv6: Dioptimasi untuk edge computing dan efisiensi tinggi.
# YOLOv7: Lebih cepat dan akurat daripada YOLOv5, sangat cocok untuk deployment real-time.
# YOLOv8 (by Ultralytics): Versi terbaru dengan dukungan:
#    - Object Detection (detect)
#    - Instance Segmentation (segment)
#    - Classification (classify)
#    - Pose Estimation (pose)
#    - OBB (oriented bounding box)
#
# YOLOv12: Generasi terbaru dari Ultralytics (2024), mendukung task yang sama seperti YOLOv8,
#          tapi lebih ringan, cepat, dan akurat (dengan arsitektur baru seperti RT-DETR).
#
# Gunakan:
#  - YOLOv8n / YOLOv12n → untuk perangkat rendah (Nano)
#  - YOLOv8s / YOLOv12s → untuk kecepatan real-time
#  - YOLOv8x / YOLOv12x → untuk akurasi maksimum (tapi lebih lambat)
#  - YOLOv8-seg / YOLOv12-seg → untuk segmentasi objek
#  - YOLOv8-pose → untuk deteksi keypoints manusia

#endregion

# Kelas yang akan dideteksi: laptop (63), handphone (67)
TARGET_CLASSES = [63, 67]

def deteksi_perangkat(filepath):
    filename = os.path.basename(filepath)

    # Baca gambar / video
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        hasil_path, jumlah = proses_gambar(filepath)
    elif filename.lower().endswith('.mp4'):
        hasil_path, jumlah = proses_video(filepath)
    else:
        return None, 0

    return hasil_path, jumlah

def proses_gambar(path):
    hasil = model(path)[0]
    img = hasil.orig_img.copy()

    jumlah = 0
    for box in hasil.boxes:
        cls_id = int(box.cls[0])
        if cls_id in TARGET_CLASSES:
            jumlah += 1
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = model.names[cls_id]
            conf = float(box.conf[0])
            cv2.rectangle(img, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    nama_hasil = "hasil_" + os.path.basename(path)
    path_hasil = os.path.join("uploads", nama_hasil)
    cv2.imwrite(path_hasil, img)

    return path_hasil, jumlah


def proses_video(path):
    cap = cv2.VideoCapture(path)
    nama_hasil = "hasil_" + os.path.basename(path)
    out_path = os.path.join("uploads", nama_hasil)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
    jumlah_total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hasil = model(frame)[0]
        img = hasil.orig_img.copy()

        for box in hasil.boxes:
            cls_id = int(box.cls[0])
            if cls_id in TARGET_CLASSES:
                jumlah_total += 1
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                label = model.names[cls_id]
                conf = float(box.conf[0])
                cv2.rectangle(img, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
                cv2.putText(img, f'{label} {conf:.2f}', (xyxy[0], xyxy[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(img)

    cap.release()
    out.release()
    return out_path, jumlah_total


def deteksi_frame(frame):
    hasil = model(frame)[0]
    img = hasil.orig_img.copy()

    for box in hasil.boxes:
        cls_id = int(box.cls[0])
        if cls_id in TARGET_CLASSES:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            label = model.names[cls_id]
            conf = float(box.conf[0])
            cv2.rectangle(img, tuple(xyxy[:2]), tuple(xyxy[2:]), (0, 255, 0), 2)
            cv2.putText(img, f'{label} {conf:.2f}', (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img

