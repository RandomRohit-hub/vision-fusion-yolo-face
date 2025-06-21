import cv2
import argparse
import os
from ultralytics import YOLO
import torch
from simple_facerec import SimpleFacerec
import face_recognition

def load_source(source):
    if source == "0":
        return True, cv2.VideoCapture(0)
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']
    if source.split('.')[-1].lower() in img_formats:
        return False, cv2.imread(source)
    return True, cv2.VideoCapture(source)

def draw_yolo_boxes(image, results, names, thickness=2):
    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{names[cls_id]} {conf:.2f}"
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), thickness)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness)
    return image

def compare_images(img1_path, img2_path):
    try:
        img1 = face_recognition.load_image_file(img1_path)
        img2 = face_recognition.load_image_file(img2_path)

        enc1 = face_recognition.face_encodings(img1)[0]
        enc2 = face_recognition.face_encodings(img2)[0]

        result = face_recognition.compare_faces([enc1], enc2)
        print(f"[Image Comparison] Match result between {img1_path} and {img2_path}: {result}")
        return result[0]
    except Exception as e:
        print("Error comparing images:", e)
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="0", help="Image/video path or '0' for webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLOv8 confidence threshold")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box thickness")
    parser.add_argument("--delay", type=int, default=45, help="Delay between frames in ms")
    parser.add_argument("--compare", nargs=2, help="Compare two face images (img1 img2)")
    args = parser.parse_args()

    # Optional: Image comparison
    if args.compare:
        match = compare_images(args.compare[0], args.compare[1])
        print("Face match:", "✅ Yes" if match else "❌ No")
        exit()

    # Device selection
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Info] Using device: {device}")

    # Load models
    model = YOLO(args.model).to(device)
    sfr = SimpleFacerec()
    sfr.load_encoding_images("images/")  # Your folder with labeled face images

    # Load source
    is_video, src = load_source(args.source)

    if not is_video:
        # If static image
        results = model(src, conf=args.conf)
        names = model.names
        img = draw_yolo_boxes(src, results, names, args.thickness)
        cv2.imshow("Detection", img)
        cv2.waitKey(0)
    else:
        # If video or webcam
        cap = src
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO object detection
            results = model(frame, conf=args.conf)
            names = model.names
            frame = draw_yolo_boxes(frame, results, names, args.thickness)

            # Face recognition
            face_locations, face_names = sfr.detect_known_faces(frame)
            for loc, name in zip(face_locations, face_names):
                y1, x2, y2, x1 = loc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.imshow("YOLO + Face Recognition", frame)
            if cv2.waitKey(args.delay) & 0xFF == 27:
                break

        cap.release()
    cv2.destroyAllWindows()
