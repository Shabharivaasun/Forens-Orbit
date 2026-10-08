#!/usr/bin/env python3
import json
import os
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

try:
    from forensorbit_detection.pdf_generator import generate_forensic_pdf_report
    PDF_GEN_AVAILABLE = True
except ImportError:
    PDF_GEN_AVAILABLE = False


class EvidenceDetectorNode(Node):
    def __init__(self):
        super().__init__('evidence_detector_node')

        # Declare parameters
        self.declare_parameter('frame_sample_interval', 15)
        self.declare_parameter('weights_path', '')
        self.declare_parameter('confidence_threshold', 0.4)
        self.declare_parameter('output_base_dir', '~/Robotics/Feature2/data/reports')

        # Create Subscriber & Publisher
        self.subscription = self.create_subscription(
            String,
            '/capture/video_path',
            self.video_path_callback,
            10
        )
        self.report_path_pub = self.create_publisher(
            String,
            '/detection/report_path',
            10
        )

        self.get_logger().info('Evidence Detector Node initialized and listening on /capture/video_path')

    def get_weights_path(self):
        param_path = self.get_parameter('weights_path').get_parameter_value().string_value
        if param_path and param_path.strip():
            target = os.path.expanduser(param_path.strip())
            if os.path.exists(target):
                return target

        # Default package share path
        pkg_share = get_package_share_directory('forensorbit_detection')
        ncnn_dir = os.path.join(pkg_share, 'weights', 'best_ncnn_model')
        
        # If NCNN directory has model files (not just .gitkeep), use it
        if os.path.exists(ncnn_dir):
            valid_files = [f for f in os.listdir(ncnn_dir) if not f.startswith('.')]
            if len(valid_files) > 0:
                return ncnn_dir

        # Fallback to standard pretrained YOLO model for immediate pipeline testing
        self.get_logger().info('No custom NCNN model found in weights/best_ncnn_model. Falling back to default "yolov8n.pt" for testing.')
        return 'yolov8n.pt'

    def video_path_callback(self, msg: String):
        video_path = os.path.expanduser(msg.data.strip())
        self.get_logger().info(f'Received video path: {video_path}')

        if not os.path.exists(video_path):
            self.get_logger().error(f'Video file does not exist: {video_path}')
            return

        if not ULTRALYTICS_AVAILABLE:
            self.get_logger().error('ultralytics package is not installed. Cannot run YOLO inference.')
            return

        weights_path = self.get_weights_path()
        self.get_logger().info(f'Loading NCNN model weights from: {weights_path}')

        try:
            model = YOLO(weights_path, task='detect')
        except Exception as e:
            self.get_logger().error(f'Failed to load model from {weights_path}: {e}')
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.get_logger().error(f'Failed to open video file: {video_path}')
            return

        sample_interval = self.get_parameter('frame_sample_interval').get_parameter_value().integer_value
        conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        output_base_dir = os.path.expanduser(
            self.get_parameter('output_base_dir').get_parameter_value().string_value
        )

        # Derive folder timestamp name from video filename
        video_filename = os.path.basename(video_path)
        folder_name = os.path.splitext(video_filename)[0]
        report_dir = os.path.abspath(os.path.join(output_base_dir, folder_name))
        os.makedirs(report_dir, exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        frame_count = 0
        processed_count = 0
        detections_list = []

        self.get_logger().info(
            f'Processing video with interval={sample_interval}, conf_thresh={conf_threshold}, fps={fps:.2f}'
        )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % sample_interval != 0:
                continue

            processed_count += 1
            orig_h, orig_w = frame.shape[:2]
            timestamp_sec = round((frame_count - 1) / fps, 2)

            # Resize frame to 416x416 for inference
            resized_frame = cv2.resize(frame, (416, 416))

            # Run inference
            results = model(resized_frame, imgsz=416, verbose=False)

            frame_detections = []
            annotated_frame = frame.copy()
            has_detection = False

            scale_x = orig_w / 416.0
            scale_y = orig_h / 416.0

            for r in results:
                boxes = r.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    conf = float(box.conf[0].cpu().numpy())
                    if conf < conf_threshold:
                        continue

                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names.get(cls_id, str(cls_id)) if hasattr(model, 'names') and model.names else str(cls_id)

                    # Scale box back to original dimensions
                    x1_416, y1_416, x2_416, y2_416 = box.xyxy[0].cpu().numpy()
                    x1 = max(0, int(x1_416 * scale_x))
                    y1 = max(0, int(y1_416 * scale_y))
                    x2 = min(orig_w, int(x2_416 * scale_x))
                    y2 = min(orig_h, int(y2_416 * scale_y))

                    # Draw bounding box and label on original frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f'{cls_name} {conf:.2f}'
                    cv2.putText(
                        annotated_frame, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

                    image_filename = f'frame_{frame_count}_det_{len(frame_detections)}.jpg'

                    det_record = {
                        'class': cls_name,
                        'confidence': round(conf, 4),
                        'frame_number': frame_count,
                        'video_timestamp_seconds': timestamp_sec,
                        'image_filename': image_filename
                    }
                    frame_detections.append(det_record)
                    has_detection = True

            if has_detection:
                for det in frame_detections:
                    img_path = os.path.join(report_dir, det['image_filename'])
                    cv2.imwrite(img_path, annotated_frame)
                    detections_list.append(det)

            if processed_count % 10 == 0:
                self.get_logger().info(
                    f'Processed {processed_count} sampled frames ({frame_count} total frames), {len(detections_list)} detections so far...'
                )

        cap.release()

        # Write report.json
        report_data = {
            'disclaimer': 'Findings are advisory, not conclusive — for investigator review.',
            'source_video': video_path,
            'detections': detections_list
        }

        report_json_path = os.path.join(report_dir, 'report.json')
        with open(report_json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=4)

        # Generate professional Forensic PDF Report
        if PDF_GEN_AVAILABLE:
            try:
                generate_forensic_pdf_report(report_dir, report_data)
                self.get_logger().info(f'Generated forensic_report.pdf inside: {report_dir}')
            except Exception as pdf_err:
                self.get_logger().warning(f'Failed to generate PDF report: {pdf_err}')

        self.get_logger().info(
            f'Processing completed! Total sampled frames: {processed_count}, total detections: {len(detections_list)}. Report saved to: {report_dir}'
        )

        # Publish report directory path
        report_msg = String()
        report_msg.data = report_dir
        self.report_path_pub.publish(report_msg)
        self.get_logger().info(f'Published report path to /detection/report_path: {report_dir}')


def main(args=None):
    rclpy.init(args=args)
    node = EvidenceDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
