import cv2
import numpy as np
import onnxruntime as ort
import os
import time

# --- CONFIGURATION ---
SEG_MODEL_PATH = "massyl/seg_models/stdc813m_maxmiou_4.onnx"
OD_MODEL_PATH  = "massyl/od_models/yolov8s_best_2.onnx"
VIDEO_SOURCE   = "massyl/data/videos/video_laf_left.mp4" 

# Model Settings
SEG_INPUT_SIZE = (1024, 512)
OD_INPUT_SIZE  = 640
CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.45

# Decision Logic
ROAD_OVERLAP_THRESHOLD = 0.1
ALERT_COOLDOWN = 5.0

class FusionPipeline:
    def __init__(self):
        self.init_models()
        self.last_alert_time = 0
        
        # Colors
        self.color_road_obs = (0, 255, 0)   # Green
        self.color_ignored  = (0, 0, 255)   # Red
        self.color_intrsct  = (255, 0, 255) # Purple (Intersection)

    def init_models(self):
        print("Initializing ONNX Sessions...")
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.seg_sess = ort.InferenceSession(SEG_MODEL_PATH, providers=providers)
        self.seg_input_name = self.seg_sess.get_inputs()[0].name
        
        self.od_sess = ort.InferenceSession(OD_MODEL_PATH, providers=providers)
        self.od_input_name = self.od_sess.get_inputs()[0].name
        self.od_output_name = self.od_sess.get_outputs()[0].name

    def preprocess_seg(self, frame):
        img = cv2.resize(frame, SEG_INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        input_tensor = img.transpose(2, 0, 1)[np.newaxis, :]
        return input_tensor

    def preprocess_od(self, frame):
        shape = frame.shape[:2]
        new_shape = (OD_INPUT_SIZE, OD_INPUT_SIZE)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = dw / 2, dh / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        else:
            img = frame

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

        input_tensor = img.transpose((2, 0, 1))[::-1]
        input_tensor = np.ascontiguousarray(input_tensor).astype(np.float32) / 255.0
        input_tensor = input_tensor[None]
        return input_tensor, (ratio, (dw, dh))

    def postprocess_od(self, output, dwdh, ratio, orig_shape):
        prediction = output[0][0]
        predictions = np.transpose(prediction, (1, 0))

        boxes, scores, class_ids = [], [], []
        pred_boxes = predictions[:, :4]
        pred_scores = predictions[:, 4:]
        max_scores = np.max(pred_scores, axis=1)
        max_ids = np.argmax(pred_scores, axis=1)

        mask = max_scores >= CONF_THRESHOLD
        filtered_boxes = pred_boxes[mask]
        filtered_scores = max_scores[mask]
        filtered_ids = max_ids[mask]

        if len(filtered_boxes) == 0: return [], [], []

        nms_boxes = []
        for box in filtered_boxes:
            cx, cy, w, h = box
            nms_boxes.append([cx - w/2, cy - h/2, w, h])

        indices = cv2.dnn.NMSBoxes(nms_boxes, filtered_scores.tolist(), CONF_THRESHOLD, IOU_THRESHOLD)

        if len(indices) > 0:
            pad_w, pad_h = dwdh
            scale_w, scale_h = ratio
            for i in indices.flatten():
                x, y, w, h = nms_boxes[i]
                x = (x - pad_w) / scale_w
                y = (y - pad_h) / scale_h
                w = w / scale_w
                h = h / scale_h
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(orig_shape[1], int(x + w))
                y2 = min(orig_shape[0], int(y + h))
                boxes.append([x1, y1, x2, y2])
                scores.append(filtered_scores[i])
                class_ids.append(filtered_ids[i])
        
        return boxes, scores, class_ids

    def check_road_overlap(self, box, road_mask):
        x1, y1, x2, y2 = box
        roi = road_mask[y1:y2, x1:x2]
        if roi.size == 0: return 0.0, None
        
        intersection_mask = (roi == 1)
        road_pixels = np.count_nonzero(intersection_mask)
        total_pixels = roi.size
        
        return road_pixels / total_pixels, intersection_mask

    def draw_performance_bars(self, frame, seg_ms, od_ms):
        """
        Draws compact latency bars in the bottom-left corner.
        """
        h, w = frame.shape[:2]
        
        # Settings for small bars
        bar_height = 8 
        bar_spacing = 15
        max_width = 100  # Max width in pixels (for ~100ms)
        start_x = 10
        start_y_seg = h - 30
        start_y_od  = h - 15
        
        # 1. Backgrounds (Dark Grey)
        cv2.rectangle(frame, (start_x, start_y_seg), (start_x + max_width, start_y_seg + bar_height), (50, 50, 50), -1)
        cv2.rectangle(frame, (start_x, start_y_od),  (start_x + max_width, start_y_od + bar_height),  (50, 50, 50), -1)
        
        # 2. Fill (1px = 1ms)
        w_seg = min(max_width, int(seg_ms))
        w_od  = min(max_width, int(od_ms))
        
        cv2.rectangle(frame, (start_x, start_y_seg), (start_x + w_seg, start_y_seg + bar_height), (255, 191, 0), -1) # Blue
        cv2.rectangle(frame, (start_x, start_y_od),  (start_x + w_od, start_y_od + bar_height),  (0, 255, 255), -1) # Yellow
        
        # 3. Text (Small font)
        font_scale = 0.4
        cv2.putText(frame, f"Seg: {seg_ms:.0f}ms", (start_x + max_width + 5, start_y_seg + bar_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)
        cv2.putText(frame, f"OD: {od_ms:.0f}ms",  (start_x + max_width + 5, start_y_od + bar_height), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), 1)

    def run(self):
        if '*' in VIDEO_SOURCE:
            from glob import glob
            files = sorted(glob(VIDEO_SOURCE))
            cap = None
            print(f"Processing {len(files)} images...")
        else:
            cap = cv2.VideoCapture(VIDEO_SOURCE)
            files = None
            print("Processing video stream...")

        idx = 0
        while True:
            if cap:
                ret, frame = cap.read()
                if not ret: break
            else:
                if idx >= len(files): break
                frame = cv2.imread(files[idx])
                idx += 1
            
            if frame is None: continue
            
            h, w = frame.shape[:2]
            
            # --- 1. SEGMENTATION ---
            t_seg_start = time.time()
            seg_in = self.preprocess_seg(frame)
            seg_out = self.seg_sess.run(None, {self.seg_input_name: seg_in})
            seg_map = np.argmax(seg_out[0][0], axis=0).astype(np.uint8)
            seg_mask_full = cv2.resize(seg_map, (w, h), interpolation=cv2.INTER_NEAREST)
            t_seg_end = time.time()

            # --- 2. OBJECT DETECTION ---
            t_od_start = time.time()
            od_in, params = self.preprocess_od(frame)
            od_out = self.od_sess.run([self.od_output_name], {self.od_input_name: od_in})
            boxes, scores, classes = self.postprocess_od(od_out, params[1], params[0], (h, w))
            t_od_end = time.time()

            # --- 3. LOGIC & VISUALIZATION ---
            alert_triggered = False
            
            # --- DISTANCE LINES (ADJUST THESE) ---
            # Horizon / Far Line (Yellow)
            line_far_y = int(h * 0.40) # 60% down from the top
            cv2.line(frame, (0, line_far_y), (w, line_far_y), (0, 255, 255), 1)
            
            # Critical / Close Line (Red)
            line_close_y = int(h * 0.85) # 85% down from the top
            cv2.line(frame, (0, line_close_y), (w, line_close_y), (0, 0, 255), 2)
            
            for i, box in enumerate(boxes):
                overlap_ratio, overlap_mask = self.check_road_overlap(box, seg_mask_full)
                
                if overlap_ratio > ROAD_OVERLAP_THRESHOLD:
                    color = self.color_road_obs
                    label = f"OBS: {overlap_ratio*100:.0f}%"
                    alert_triggered = True
                    
                    # Intersection Highlight (Purple)
                    roi_color = np.zeros((box[3]-box[1], box[2]-box[0], 3), dtype=np.uint8)
                    roi_color[overlap_mask] = self.color_intrsct 
                    
                    frame_roi = frame[box[1]:box[3], box[0]:box[2]]
                    mask_bool = overlap_mask.astype(bool)
                    if mask_bool.any():
                        frame_roi[mask_bool] = cv2.addWeighted(frame_roi[mask_bool], 0.5, roi_color[mask_bool], 0.5, 0)
                else:
                    color = self.color_ignored
                    label = f"IGN: {overlap_ratio*100:.0f}%"

                # Draw Box
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
                cv2.putText(frame, label, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # --- 4. GLOBAL UPDATES ---
            if alert_triggered:
                if (time.time() - self.last_alert_time) > ALERT_COOLDOWN:
                    print(f"!!! CRITICAL OBSTACLE !!! Frame {idx}")
                    cv2.circle(frame, (30, 30), 15, (0, 0, 255), -1) 
                    self.last_alert_time = time.time()

            # Transparent Road Overlay
            color_mask = np.zeros_like(frame)
            color_mask[seg_mask_full == 1] = [0, 255, 0]
            frame = cv2.addWeighted(frame, 1.0, color_mask, 0.2, 0)

            # Latency Bars
            seg_ms = (t_seg_end - t_seg_start) * 1000
            od_ms  = (t_od_end - t_od_start) * 1000
            self.draw_performance_bars(frame, seg_ms, od_ms)

            cv2.imshow("Fusion ADAS", frame)
            
            key = cv2.waitKey(1 if cap else 0)
            if key == ord('q'): break
            if not cap and key == ord('d'): continue 

        if cap: cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    pipeline = FusionPipeline()
    pipeline.run()