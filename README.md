
```markdown
# YOLO + Face Recognition (Beta)

🚧 **Beta Release** — still under active development. Expect bugs or incomplete features.

This project combines **YOLOv8 object detection** and **real-time face recognition** using `face_recognition` for multi-modal AI-powered computer vision.

---

## ✨ Features

- ✅ **YOLOv8 object detection** (image, video, or webcam)
- ✅ **Real-time face recognition** using your own face image dataset
- ✅ **Image-to-image face comparison** (via CLI)
- 🖼️ Flexible input: webcam, image, video file
- ⚙️ Built-in argument parser for easy customization
- 🧠 CUDA support (runs on GPU if available)

---

## 📁 Folder Structure

```

project-root/
│
├── main.py                 # Main script for detection
├── simple\_facerec.py       # Face recognition helper class
├── yolov8n.pt              # YOLOv8 model file (object or face detection)
├── images/                 # Folder containing known face images (used for recognition)
│   ├── Elon Musk.jpg
│   └── messi.jpg
└── README.md

````

---

## 📦 Installation

```bash
git clone https://github.com/your-username/yolo-face-detection-beta.git
cd yolo-face-detection-beta

pip install -r requirements.txt
````

Create a `requirements.txt` file with:

```
opencv-python
face_recognition
ultralytics
numpy
torch
```

---

## 🚀 Usage

### 1. 📹 Real-time detection from webcam:

```bash
python main.py --source 0 --model yolov8n.pt
```

### 2. 🖼 Detect from image:

```bash
python main.py --source "input.jpg"
```

### 3. 🎞 Detect from video:

```bash
python main.py --source "video.mp4"
```

### 4. 🧑‍🤝‍🧑 Compare two face images:

```bash
python main.py --compare "images/Elon Musk.jpg" "images/messi.jpg"
```

---

## 🧠 Known Face Images Format

* Put labeled images in the `images/` folder
* File name becomes the label shown (e.g. `Bill Gates.jpg` → "Bill Gates")

---

## ⚠️ Beta Notice

This is a **beta version**. Bugs, performance issues, or limitations may exist. Contributions and bug reports are welcome!

---

## 📜 License

MIT License

---

## 🙋‍♂️ Author

Developed by [Your Name](https://github.com/your-username)

```

--

Let me know if you wa

- A sample `requirements.txt` 
- A GitHub-ready repo zip
- Auto-save of results (bounding boxes or CSV logs)


```
