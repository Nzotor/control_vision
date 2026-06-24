import cv2
import time
import os
from ultralytics import YOLO

def benchmark_single_model(model_path, video_source, num_frames=100):
    print(f"\n{'-'*50}\n[ARCHI 1] SINGLE MODEL (16 Classes)\nModel: {model_path}\n{'-'*50}")
    
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"[ERROR] Failed to load {model_path}: {e}")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Video {video_source} not found.")
        return
    
    # Warm-up phase
    print("[INFO] Warming up...")
    for _ in range(5):
        ret, frame = cap.read()
        if ret: model.predict(cv2.resize(frame, (416, 416)), verbose=False)
            
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"[INFO] Running inference on {num_frames} frames...")
    start_time = time.time()
    frames_processed = 0
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (416, 416))
        _ = model.predict(frame_resized, verbose=False)
        frames_processed += 1

    total_time = time.time() - start_time
    cap.release()
    
    afficher_resultats(frames_processed, total_time)


def benchmark_two_models(det_path, cls_path, video_source, num_frames=100):
    print(f"\n{'-'*50}\n[ARCHI 2] CASCADE (Detector + Classifier)\nModels: {det_path} & {cls_path}\n{'-'*50}")
    
    try:
        modele_det = YOLO(det_path, task='detect')
        modele_cls = YOLO(cls_path, task='classify')
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"[ERROR] Video {video_source} not found.")
        return
    
    # Warm-up phase
    print("[INFO] Warming up...")
    for _ in range(5):
        ret, frame = cap.read()
        if ret: modele_det.predict(cv2.resize(frame, (416, 416)), verbose=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    print(f"[INFO] Running inference on {num_frames} frames...")
    start_time = time.time()
    frames_processed = 0
    
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret: break
        
        frame_resized = cv2.resize(frame, (416, 416))
        
        # 1. DETECTION
        res_det = modele_det.predict(frame_resized, verbose=False)[0]
        boxes = res_det.boxes.xyxy.cpu().numpy()
        
        # 2. MEGA-BOX & CLASSIFICATION
        if len(boxes) > 0:
            x_min, y_min, x_max, y_max = boxes[0] 
            
            # Dynamic margin 15%
            w_box = x_max - x_min
            h_box = y_max - y_min
            marge_x = int(w_box * 0.15)
            marge_y = int(h_box * 0.15)
            
            x1_safe = max(0, int(x_min - marge_x))
            y1_safe = max(0, int(y_min - marge_y))
            x2_safe = min(frame_resized.shape[1], int(x_max + marge_x))
            y2_safe = min(frame_resized.shape[0], int(y_max + marge_y))
            
            crop = frame_resized[y1_safe:y2_safe, x1_safe:x2_safe]
            
            if crop.size > 0:
                _ = modele_cls.predict(cv2.resize(crop, (224, 224)), verbose=False)

        frames_processed += 1

    total_time = time.time() - start_time
    cap.release()
    
    afficher_resultats(frames_processed, total_time)


def afficher_resultats(frames, temps):
    if frames == 0:
        print("[ERROR] No frames processed.")
        return
        
    fps = frames / temps
    latence_ms = (temps / frames) * 1000
    print(f">>> RESULTS <<<")
    print(f"Total time : {temps:.2f} s")
    print(f"Speed      : {fps:.2f} FPS")
    print(f"Latency    : {latence_ms:.2f} ms/frame")


if __name__ == "__main__":
    VIDEO = "underwater_test_video.mp4"
    FRAMES_TO_TEST = 200
    
    # 1. Define the formats to test
    formats = [
        {"name": "PyTorch", "suffix": ".pt"},
        {"name": "ONNX", "suffix": ".onnx"},
        {"name": "NCNN", "suffix": "_ncnn_model"} 
    ]
    
    # 2. Base names of models
    base_name_single = "best_modele_unique"
    base_name_det = "best_detecteur"
    base_name_cls = "best_classifieur"

    print(f"\n{'='*60}")
    print("🚀 ULTIMATE BENCHMARK: 1-MODEL vs 2-MODELS ACROSS FORMATS")
    print(f"{'='*60}\n")

    for fmt in formats:
        print(f"\n\n{'#'*60}")
        print(f"||| TESTING FORMAT : {fmt['name'].upper()} |||")
        print(f"{'#'*60}")
        
        ext = fmt["suffix"]
        
        # Test 1: Single Model Architecture
        path_single = f"{base_name_single}{ext}"
        if os.path.exists(path_single):
            benchmark_single_model(path_single, VIDEO, num_frames=FRAMES_TO_TEST)
        else:
            print(f"\n[WARNING] Skipping Archi 1: {path_single} not found.")

        # Test 2: Cascade Architecture
        path_det = f"{base_name_det}{ext}"
        path_cls = f"{base_name_cls}{ext}"
        if os.path.exists(path_det) and os.path.exists(path_cls):
            benchmark_two_models(path_det, path_cls, VIDEO, num_frames=FRAMES_TO_TEST)
        else:
            print(f"\n[WARNING] Skipping Archi 2: {path_det} or {path_cls} not found.")