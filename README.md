# Deteksi-Laptop-dan-Smartphone-menggunakan-YOLOv12-OpenCV-Flask

Aplikasi web yang sederhana memungkinkan deteksi keberadaan **Laptop** dan **Smartphone** dalam gambar, video, dan kamera real-time menggunakan **YOLOv12**, **OpenCV**, dan **Flask**.

---

## 🎯 Fitur

✅ Upload gambar dan video untuk dideteksi  
✅ Deteksi prangkat secara real-time dari kamera laptop/server  
✅ Tampilkan jumlah perangkat yang terdeteksi  
✅ Tampilan web responsif dan modern (Bootstrap 5)  
✅ Bisa diakses dari jaringan lokal (LAN)

---

## 🧑‍💻 Teknologi yang Digunakan

| Komponen       | Teknologi                     |
|----------------|-------------------------------|
| Backend        | Python 3 + Flask              |
| Deteksi AI     | YOLOv12       |
| Pengolahan Citra | OpenCV, NumPy               |
| Tampilan Web   | HTML, Bootstrap, Jinja2       |

---

📂 Struktur Folder
```csharp
├── app.py
├── templates/
│   ├── base.html
│   ├── index.html
│   └── result.html
├── static/
│   └── style.css
├── uploads/                ← Menyimpan hasil deteksi
├── utils/
│   └── detector.py         ← Deteksi YOLOv12
└── requirements.txt
```

---

## 🚀 Cara Menjalankan Proyek

### 📦 1. Clone Repo
```bash

git clone https://github.com/KANGGAFH/Deteksi-Laptop-dan-Smartphone-menggunakan-YOLOv12-OpenCV-Flask.git
cd Deteksi-Laptop-dan-Smartphone-menggunakan-YOLOv12-OpenCV-Flask
```

### 📥 2. Install Semua Dependency
```bash
pip install -r requirements.txt
```

### 🧠 3. Download Model YOLOv12
jalankan python:
```python
from ultralytics import YOLO
YOLO("yolo12x.pt")
```
Model ini akan otomatis disimpan dan digunakan untuk deteksi perangkat.

### 🟢 5. Jalankan Aplikasi
```bash
python app.py
```
Nantinya akan muncul:
```csharp
 * Running on http://127.0.0.1:5000
 * Running on http://<IP_KOMPUTER_KAMU>:5000
```

### 🌐 6. Akses dari Browser
- Lokal:
  ```arduino
  http://localhost:5000
  ```
- Perangkat Lain (masih satu jaringan)
  ```cpp
  http://<IP_KOMPUTER_KAMU>:5000
  ```
Cek IP lokal dengan:
```bash
ipconfig  # Windows
ifconfig  # Linux/macOS
```



