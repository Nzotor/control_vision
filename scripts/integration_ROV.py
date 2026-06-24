import cv2
import time
from collections import deque, Counter
from ultralytics import YOLO

# ==========================================
# SYSTEM CONFIGURATION
# ==========================================
BUFFER_SIZE = 15           # Number of frames to keep in memory
CONFIDENCE_THRESHOLD = 0.7 # 70% of frames in buffer must agree 

class TemporalFilter:
    def __init__(self, size, threshold_ratio):
        self.buffer = deque(maxlen=size)
        self.threshold = int(size * threshold_ratio)

    def add_prediction(self, prediction):
        self.buffer.append(prediction)

    def get_stable_gesture(self):
        if len(self.buffer) < self.buffer.maxlen:
            return None 

        counts = Counter(self.buffer)
        majority_gesture, occurrences = counts.most_common(1)[0]

        if majority_gesture is not None and occurrences >= self.threshold:
            return majority_gesture
            
        return None

    def clear(self):
        self.buffer.clear()


# ==========================================
# MAIN LOOP (INTEGRATION & STATE MACHINE)
# ==========================================
def main():
    print("[INFO] Loading YOLO model...")
    model = YOLO('best_modele_unique.onnx', task='detect')
    
    temporal_filter = TemporalFilter(size=BUFFER_SIZE, threshold_ratio=CONFIDENCE_THRESHOLD)
    
    # --- State Machine Variables ---
    is_listening = False
    session_commands = []
    last_validated_gesture = None
    
    # Replace 0 with camera index or video file
    cap = cv2.VideoCapture("underwater_test_video.mp4") 
    
    if not cap.isOpened():
        print("[ERROR] Could not open video source.")
        return

    print("\n" + "="*50)
    print("ROBOT STANDBY: Waiting for 'START_COMM' gesture...")
    print("="*50 + "\n")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
            
        # Inference
        results = model.predict(frame, verbose=False)[0]
        raw_prediction = None
        
        if len(results.boxes) > 0:
            class_id = int(results.boxes.cls[0].item())
            raw_prediction = results.names[class_id]
        
        # Feed the Temporal Filter 
        temporal_filter.add_prediction(raw_prediction)
        
        # Get the stable gesture 
        stable_gesture = temporal_filter.get_stable_gesture()
        
        # ==========================================
        # STATE MACHINE LOGIC
        # ==========================================
        if stable_gesture:
            
            # CASE A: We are NOT listening, waiting for the wake-up word
            if not is_listening:
                if stable_gesture == "start_comm":
                    is_listening = True
                    session_commands = []
                    last_validated_gesture = None
                    print("\nSTART_COMM RECEIVED: Robot is now LISTENING")
                    temporal_filter.clear() 
            
            # CASE B: We ARE listening
            else:
                if stable_gesture == "end_comm":
                    is_listening = False
                    print("\nEND_COMM RECEIVED: Robot goes back to STANDBY")
                    # --- PRINT THE FULL SESSION SUMMARY ---
                    print("-" * 50)
                    if len(session_commands) > 0:
                        sequence = " -> ".join(session_commands).upper()
                        print(f"SEQUENCE DETECTED: {sequence}")
                    else:
                        print("SEQUENCE DETECTED: [EMPTY SESSION]")
                    print("-" * 50 + "\n")
                    temporal_filter.clear()
                    
                # It's a regular command (boat, up, down, etc.)
                elif stable_gesture != "start_comm":
                    
                    # Apply Cooldown to avoid printing the same command multiple times in a row
                    if stable_gesture != last_validated_gesture:
                        session_commands.append(stable_gesture)
                        print(f"      ---> ACTION DETECTED : [{stable_gesture.upper()}]")
                        last_validated_gesture = stable_gesture
                        temporal_filter.clear()

        # Debug Display
        status_color = (0, 255, 0) if is_listening else (0, 0, 255)
        status_text = "LISTENING" if is_listening else "STANDBY"
        
        display_frame = results.plot()
        cv2.putText(display_frame, f"STATE: {status_text}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)
                    
        cv2.imshow("AUV Vision", display_frame)
        
# --- COMMANDES CLAVIER (DEBUG MODE) ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
            
        elif key == ord('s') and not is_listening:
            # Force le début de communication avec la touche 'S'
            is_listening = True
            session_commands = [] 
            print("\nMANUAL OVERRIDE: Robot is now LISTENING")
            temporal_filter.clear()
            
        elif key == ord('e') and is_listening:
            # Force la fin de communication avec la touche 'E'
            is_listening = False
            print("\nMANUAL OVERRIDE: Robot goes back to STANDBY")
            print("-" * 50)
            if len(session_commands) > 0:
                print(f"SEQUENCE DETECTED: {' -> '.join(session_commands).upper()}")
            else:
                print("SEQUENCE DETECTED: [EMPTY SESSION]")
            print("-" * 50 + "\n")
            temporal_filter.clear()
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()