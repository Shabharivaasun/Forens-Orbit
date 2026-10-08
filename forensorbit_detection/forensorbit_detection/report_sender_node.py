#!/usr/bin/env python3
import os
import subprocess
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class ReportSenderNode(Node):
    def __init__(self):
        super().__init__('report_sender_node')

        # Declare ROS2 parameters
        self.declare_parameter('ssh_host', '')
        self.declare_parameter('ssh_user', '')
        self.declare_parameter('ssh_key_path', '~/.ssh/id_rsa')
        self.declare_parameter('remote_report_path', '~/Desktop/forensic_reports')
        self.declare_parameter('transfer_method', 'rsync')
        self.declare_parameter('max_retries', 3)
        self.declare_parameter('retry_delay_sec', 5.0)

        # Create Subscriber & Publisher
        self.subscription = self.create_subscription(
            String,
            '/detection/report_path',
            self.report_path_callback,
            10
        )
        self.status_pub = self.create_publisher(
            String,
            '/transfer/report_status',
            10
        )

        self.get_logger().info('Report Sender Node initialized and listening on /detection/report_path')

    def report_path_callback(self, msg: String):
        report_dir = os.path.expanduser(msg.data.strip())
        self.get_logger().info(f'Received report path to transfer: {report_dir}')

        if not os.path.exists(report_dir):
            self.get_logger().error(f'Report directory does not exist: {report_dir}')
            self.publish_status('failed')
            return

        ssh_host = self.get_parameter('ssh_host').get_parameter_value().string_value.strip()
        ssh_user = self.get_parameter('ssh_user').get_parameter_value().string_value.strip()
        ssh_key_path = os.path.expanduser(
            self.get_parameter('ssh_key_path').get_parameter_value().string_value.strip()
        )
        remote_report_path = self.get_parameter('remote_report_path').get_parameter_value().string_value.strip()
        transfer_method = self.get_parameter('transfer_method').get_parameter_value().string_value.strip().lower()
        max_retries = self.get_parameter('max_retries').get_parameter_value().integer_value
        retry_delay_sec = self.get_parameter('retry_delay_sec').get_parameter_value().double_value

        if not ssh_host or not ssh_user:
            self.get_logger().error('ssh_host or ssh_user parameters are not configured properly.')
            self.publish_status('failed')
            return

        # Read report.json to find original source_video path for complete transfer
        source_video = None
        report_json_path = os.path.join(report_dir, 'report.json')
        if os.path.exists(report_json_path):
            try:
                import json
                with open(report_json_path, 'r', encoding='utf-8') as f:
                    rdata = json.load(f)
                    src_v = rdata.get('source_video')
                    if src_v and os.path.exists(src_v):
                        source_video = src_v
            except Exception as ex:
                self.get_logger().warning(f'Could not read source_video from report.json: {ex}')

        items_to_send = [report_dir]
        if source_video:
            items_to_send.append(source_video)
            self.get_logger().info(f'Including complete original video in transfer: {source_video}')

        destination = f'{ssh_user}@{ssh_host}:{remote_report_path}/'
        if transfer_method == 'rsync':
            ssh_cmd = f'ssh -i {ssh_key_path} -o StrictHostKeyChecking=no -o ConnectTimeout=10'
            transfer_cmd = ['rsync', '-avz', '-e', ssh_cmd, *items_to_send, destination]
        elif transfer_method == 'scp':
            # Windows OpenSSH includes scp but generally does not include rsync.
            transfer_cmd = [
                'scp', '-r', '-i', ssh_key_path,
                '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10',
                *items_to_send, destination
            ]
        else:
            self.get_logger().error('transfer_method must be "rsync" or "scp".')
            self.publish_status('failed')
            return

        success = False
        for attempt in range(1, max_retries + 1):
            self.get_logger().info(
                f'Attempt {attempt}/{max_retries}: Uploading report folder "{report_dir}" to {ssh_user}@{ssh_host}:{remote_report_path}/...'
            )

            try:
                result = subprocess.run(
                    transfer_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=120
                )

                if result.returncode == 0:
                    self.get_logger().info(f'Successfully transferred report folder on attempt {attempt}.')
                    success = True
                    break
                else:
                    self.get_logger().warning(
                        f'Attempt {attempt} failed (return code {result.returncode}): {result.stderr.strip()}'
                    )
            except Exception as e:
                self.get_logger().warning(f'Attempt {attempt} encountered exception: {e}')

            if attempt < max_retries:
                self.get_logger().info(f'Retrying in {retry_delay_sec} seconds...')
                time.sleep(retry_delay_sec)

        final_status = 'success' if success else 'failed'
        if success:
            self.get_logger().info(f'Final transfer status for "{report_dir}": SUCCESS')
        else:
            self.get_logger().error(f'Final transfer status for "{report_dir}": FAILED after {max_retries} attempts')

        self.publish_status(final_status)

    def publish_status(self, status: str):
        msg = String()
        msg.data = status
        self.status_pub.publish(msg)
        self.get_logger().info(f'Published status to /transfer/report_status: {status}')


def main(args=None):
    rclpy.init(args=args)
    node = ReportSenderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
