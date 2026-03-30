"""
Overview: Loads a pre-trained YOLO model to perform object detection on a video file.
It filters detections by a confidence threshold, draws custom crosshairs
on the targets, and exports the processed video.
"""

import cv2
from ultralytics import YOLO


def main():

    print("Loading model...")
    model = YOLO('assets/best.pt')


    video_path = 'assets/test_video.mp4'
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Video file not found. Please check the path.")
        return

    # output video settings
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    output_path = 'processed_video.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video, please wait...")

    # frame by frame processsing
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Exit when the video ends

        # Run YOLO inference
        results = model(frame, verbose=False)

        for result in results:
            for box in result.boxes:
                # Get coordinates, class ID, and confidence score
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Ignore low-confidence detections to prevent false positives
                if conf < 0.40:
                    continue

                # Calculate target center
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                # drawing the cross-hair
                color = (0, 255, 0)  # Green (BGR format)
                thickness = 2
                line_len = 15

                # Red center dot
                cv2.circle(frame, (center_x, center_y), 3, (0, 0, 255), -1)

                # Top, Bottom, Left, Right lines
                cv2.line(frame, (center_x, center_y - 5), (center_x, center_y - 5 - line_len), color, thickness)
                cv2.line(frame, (center_x, center_y + 5), (center_x, center_y + 5 + line_len), color, thickness)
                cv2.line(frame, (center_x - 5, center_y), (center_x - 5 - line_len, center_y), color, thickness)
                cv2.line(frame, (center_x + 5, center_y), (center_x + 5 + line_len, center_y), color, thickness)

                # Draw label and confidence score
                label = f"{model.names[cls_id]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write and display the processed frame
        out.write(frame)
        cv2.imshow('object with cross-hair', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Process complete! Video saved to: {output_path}")


if __name__ == '__main__':
    main()