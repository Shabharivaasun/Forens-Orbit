#!/usr/bin/env python3
"""Record a camera clip and hand it to the ForensOrbit detection pipeline."""

import os
import re
import threading
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CameraCaptureNode(Node):
    """Records from an OpenCV camera device after a ``start`` command.

    Publish ``start`` or ``start <seconds>`` on ``/capture/record_command`` to
    begin a clip. Publish ``stop`` to finish it early. The completed video path
    is published on ``/capture/video_path``, which starts evidence detection.
    """

    def __init__(self):
        super().__init__('camera_capture_node')

        self.declare_parameter('camera_index', 0)
        self.declare_parameter('capture_duration_sec', 30.0)
        self.declare_parameter('fps', 20.0)
        self.declare_parameter('frame_width', 1280)
        self.declare_parameter('frame_height', 720)
        self.declare_parameter('output_dir', '/root/ros2_ws/videos')
        self.declare_parameter('filename_prefix', 'evidence')
        self.declare_parameter('codec', 'mp4v')
        self.declare_parameter('auto_start', False)

        self.video_path_pub = self.create_publisher(String, '/capture/video_path', 10)
        self.command_sub = self.create_subscription(
            String, '/capture/record_command', self.command_callback, 10
        )
        self._recording = False
        self._stop_requested = threading.Event()
        self._lock = threading.Lock()

        self.get_logger().info(
            'Camera Capture Node ready. Publish "start" or "start <seconds>" '
            'to /capture/record_command.'
        )
        if self.get_parameter('auto_start').value:
            self.start_recording()

    def command_callback(self, msg: String):
        command = msg.data.strip().lower()
        if command == 'stop':
            if self._recording:
                self._stop_requested.set()
                self.get_logger().info('Stopping current recording after the current frame.')
            else:
                self.get_logger().warning('No recording is active.')
            return

        match = re.fullmatch(r'start(?:\s+(\d+(?:\.\d+)?))?', command)
        if not match:
            self.get_logger().warning('Unknown command. Use "start", "start <seconds>", or "stop".')
            return

        duration = float(match.group(1)) if match.group(1) else None
        self.start_recording(duration)

    def start_recording(self, duration=None):
        with self._lock:
            if self._recording:
                self.get_logger().warning('A recording is already active.')
                return
            self._recording = True
            self._stop_requested.clear()
        thread = threading.Thread(target=self.record_video, args=(duration,), daemon=True)
        thread.start()

    def record_video(self, requested_duration):
        cap = None
        writer = None
        try:
            camera_index = int(self.get_parameter('camera_index').value)
            configured_fps = float(self.get_parameter('fps').value)
            width = int(self.get_parameter('frame_width').value)
            height = int(self.get_parameter('frame_height').value)
            duration = requested_duration or float(self.get_parameter('capture_duration_sec').value)
            output_dir = os.path.expanduser(str(self.get_parameter('output_dir').value))
            prefix = str(self.get_parameter('filename_prefix').value)
            codec = str(self.get_parameter('codec').value)

            if len(codec) != 4:
                raise ValueError('codec must contain exactly four characters, e.g. mp4v')
            os.makedirs(output_dir, exist_ok=True)
            filename = f'{prefix}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.mp4'
            video_path = os.path.abspath(os.path.join(output_dir, filename))

            cap = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
            if not cap.isOpened():
                # Some systems do not support an explicit V4L2 backend.
                cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                raise RuntimeError(
                    f'Cannot open camera index {camera_index}. In Docker, pass /dev/video0 '
                    'to the container, or use a Pi Camera capture adapter.'
                )

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, configured_fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS) or configured_fps
            if actual_fps <= 0:
                actual_fps = configured_fps
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height

            writer = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*codec),
                actual_fps,
                (actual_width, actual_height),
            )
            if not writer.isOpened():
                raise RuntimeError(f'Cannot create video file: {video_path}')

            max_frames = max(1, int(duration * actual_fps))
            frames_written = 0
            self.get_logger().info(
                f'Recording up to {duration:.1f}s from camera {camera_index} into {video_path}'
            )
            while frames_written < max_frames and not self._stop_requested.is_set():
                ok, frame = cap.read()
                if not ok:
                    self.get_logger().warning('Camera frame read failed; ending recording.')
                    break
                writer.write(frame)
                frames_written += 1

            if frames_written == 0:
                raise RuntimeError('No frames were captured; no video will be analysed.')

            self.get_logger().info(f'Saved {frames_written} frames to {video_path}')
            result = String()
            result.data = video_path
            self.video_path_pub.publish(result)
            self.get_logger().info('Published video path to /capture/video_path.')
        except Exception as exc:
            self.get_logger().error(f'Camera recording failed: {exc}')
        finally:
            if writer is not None:
                writer.release()
            if cap is not None:
                cap.release()
            with self._lock:
                self._recording = False
            self._stop_requested.clear()


def main(args=None):
    rclpy.init(args=args)
    node = CameraCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
