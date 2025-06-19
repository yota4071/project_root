import cv2
from ultralytics import YOLO

# YOLOモデルの読み込み（人物検出に最適なモデルに置き換えてもOK）
model = YOLO("yolo11n.pt")  # 実際には yolov11n.pt などに変更しても可

# Webカメラの起動
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 物体検出
    results = model(frame)

    # 人物だけを抽出
    boxes = results[0].boxes
    person_boxes = []
    for box in boxes:
        cls = int(box.cls[0])  # クラスIDを整数に変換
        if cls == 0:  # COCOにおける "person" は class_id = 0
            person_boxes.append(box)

    # 人物だけを描画
    frame_copy = frame.copy()
    for box in person_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_copy, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 表示
    cv2.imshow("Person Detection Only", frame_copy)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()