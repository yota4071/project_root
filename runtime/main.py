import cv2
import time
from detectors.yolo_deepsort import YOLODeepSORTTracker
from tracking.trajectory_manager import update_trajectory, get_trajectory, save_trajectory
from tracking.zone_assigner import get_zone
from tracking.zone_definitions import ZONES
from datetime import datetime

ZONE_COLORS = {
    "zone_A": (255, 0, 0),
    "zone_B": (0, 255, 0),
    "zone_C": (0, 0, 255),
}

# tracker = YOLODeepSORTTracker("models/yolo11n.pt")
tracker = YOLODeepSORTTracker("models/yolov8n.pt")
cap = cv2.VideoCapture(0)

start_time = time.time()
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    for zone_name, ((x1, y1), (x2, y2)) in ZONES.items():
        color = ZONE_COLORS.get(zone_name, (200, 200, 200))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, zone_name, (x1 + 5, y1 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    tracks = tracker.update(frame)

    for track in tracks:
        track_id = track.track_id
        x1, y1, x2, y2 = map(int, track.to_ltrb())  # bbox取得

        foot_x = int((x1 + x2) / 2)
        foot_y = y2  # 足元座標

        # update_trajectory(f"person_{track_id}", foot_x, foot_y)
        timestamp = datetime.now().isoformat()
        zone_name = get_zone(foot_x, foot_y)
        update_trajectory(f"person_{track_id}", foot_x, foot_y, zone_name, timestamp)

        # 描画
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(frame, (foot_x, foot_y), 4, (255, 0, 0), -1)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 軌跡描画（全員分）
    for tid in tracker.active_ids():
        traj = get_trajectory(f"person_{tid}")
        for i in range(1, len(traj)):
            cv2.line(frame, traj[i - 1], traj[i], (255, 0, 255), 2)

    # FPS描画
    frame_count += 1
    elapsed = time.time() - start_time
    fps = frame_count / elapsed
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLO + DeepSORT Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
save_trajectory()
cv2.destroyAllWindows()