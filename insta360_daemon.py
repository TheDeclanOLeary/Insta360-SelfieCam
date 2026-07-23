import argparse
import subprocess
import threading
import time
import sys

import av
import cv2
import requests
from insta360.rtmp import Client

# --- CONFIGURATION ---
CAMERA_SSID = "X5 135VYD.OSC"
CAMERA_IP = "192.168.42.1"
WIFI_PROFILE_NAME = "Insta360"
# The Wi-Fi interface can be passed as an argument or have a sensible default.
# For many Linux systems, it might be wlan0, but for USB dongles, it's unpredictable.
# Let's make it a required argument to be safe.

class Insta360Daemon:
    def __init__(self, client_ip: str, port: int, wifi_interface: str):
        self.client_ip = client_ip
        self.port = port
        self.wifi_interface = wifi_interface
        self.width = 1152  # Cropped width from 'selfie' app
        self.height = 864  # Cropped height
        
        self._rtmp_client = Client()
        self._codec = av.CodecContext.create('h264', 'r')
        self._running = True
        self._ffmpeg_process = None

        self._rtmp_client.on_video_stream(wait=True)(self._process_live_frame)

    def _print_status(self, message: str):
        print(f"[Insta360 Daemon] {message}", file=sys.stderr)

    def run(self):
        """Main execution flow for the daemon."""
        try:
            self._print_status("Starting...")
            self._toggle_camera_radio("up")
            
            self._print_status("Initializing FFmpeg pipeline...")
            self._start_ffmpeg_pipeline()
            
            self._print_status("Starting RTMP preview stream...")
            self._rtmp_client.open()
            self._rtmp_client.start_preview_stream()
            
            self._print_status("Daemon is up and streaming.")
            while self._running:
                time.sleep(1)

        except Exception as e:
            self._print_status(f"FATAL ERROR: {e}")
        finally:
            self._print_status("Shutting down...")
            self._running = False
            try:
                self._rtmp_client.close()
            except Exception:
                pass
            if self._ffmpeg_process:
                self._ffmpeg_process.kill()
            self._toggle_camera_radio("down")
            self._print_status("Shutdown complete.")

    def _start_ffmpeg_pipeline(self):
        """Starts the FFmpeg subprocess to stream frames over SRT."""
        ffmpeg_cmd = [
            'ffmpeg',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.width}x{self.height}',
            '-r', '24', # Assume 24fps, can be adjusted
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-tune', 'zerolatency',
            '-pix_fmt', 'yuv420p',
            '-f', 'mpegts',
            f'srt://{self.client_ip}:{self.port}?mode=caller'
        ]
        
        self._ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self._print_status(f"FFmpeg process started, streaming to srt://{self.client_ip}:{self.port}")

    async def _process_live_frame(self, **kwargs):
        """Callback that receives raw H.264 packets, decodes, and pipes to FFmpeg."""
        if not self._running or self._ffmpeg_process is None:
            return

        content = kwargs.get('content')
        if not content:
            return

        try:
            packets = self._codec.parse(content)
            for packet in packets:
                frames = self._codec.decode(packet)
                for frame in frames:
                    if not self._running:
                        return
                    
                    img_matrix = frame.to_ndarray(format='bgr24')
                    
                    # Apply live spatial cropping from selfie app
                    h, w, _ = img_matrix.shape
                    lens_w = w // 2
                    front_lens = img_matrix[:, 0:lens_w]
                    cx, cy = lens_w // 2, h // 2
                    crop_w = int(lens_w * 0.60)
                    crop_h = int(crop_w * 0.75)
                    y1, y2 = cy - (crop_h // 2), cy + (crop_h // 2)
                    x1, x2 = cx - (crop_w // 2), cx + (crop_w // 2)
                    cropped_frame = front_lens[y1:y2, x1:x2]

                    # Resize to the expected dimensions if necessary
                    if cropped_frame.shape[1] != self.width or cropped_frame.shape[0] != self.height:
                        cropped_frame = cv2.resize(cropped_frame, (self.width, self.height))

                    if self._ffmpeg_process.stdin:
                        try:
                            self._ffmpeg_process.stdin.write(cropped_frame.tobytes())
                        except (IOError, BrokenPipeError):
                            self._print_status("FFmpeg stdin broke. Stopping stream.")
                            self._running = False
                            return
        except Exception as e:
            self._print_status(f"RTMP Decode/FFmpeg Error: {e}")

    def _toggle_camera_radio(self, state: str):
        """Manages the Wi-Fi connection to the camera."""
        if state == "up":
            self._print_status(f"Disconnecting from other networks on {self.wifi_interface}...")
            subprocess.run(["nmcli", "device", "disconnect", self.wifi_interface], check=True, timeout=10)
            time.sleep(1)

            self._print_status(f"Scanning for camera network '{CAMERA_SSID}'...")
            for i in range(5):
                try:
                    subprocess.run(["nmcli", "device", "wifi", "rescan"], check=True, timeout=10)
                    time.sleep(2)
                    scan_results = subprocess.run(["nmcli", "-t", "-f", "SSID", "device", "wifi", "list"], capture_output=True, text=True, check=True)
                    if CAMERA_SSID in scan_results.stdout:
                        self._print_status("Camera found.")
                        break
                except subprocess.CalledProcessError as e:
                    self._print_status(f"Scan attempt {i+1} failed: {e.stderr}")
                    time.sleep(1)
            else:
                raise Exception(f"Camera SSID '{CAMERA_SSID}' not found after multiple scans.")

            self._print_status(f"Connecting to '{WIFI_PROFILE_NAME}'...")
            subprocess.run(["nmcli", "connection", "up", WIFI_PROFILE_NAME], check=True, timeout=35)
            
            self._print_status("Verifying camera API readiness...")
            for _ in range(10):
                try:
                    res = requests.get(f"http://{CAMERA_IP}/osc/info", timeout=2)
                    if res.status_code == 200:
                        self._print_status("Camera API is ready.")
                        return
                except requests.RequestException:
                    pass
                time.sleep(1)
            raise Exception("Camera API did not become ready in time.")

        elif state == "down":
            self._print_status(f"Disconnecting from '{WIFI_PROFILE_NAME}'...")
            subprocess.run(["nmcli", "connection", "down", WIFI_PROFILE_NAME], timeout=10, check=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insta360 to SRT streaming daemon.")
    parser.add_argument("--client-ip", required=True, help="The IP address of the client to stream to.")
    parser.add_argument("--port", type=int, default=7003, help="The SRT port to stream to.")
    parser.add_argument("--wifi-interface", required=True, help="The name of the Wi-Fi interface to use (e.g., wlan0).")
    
    args = parser.parse_args()
    
    daemon = Insta360Daemon(client_ip=args.client_ip, port=args.port, wifi_interface=args.wifi_interface)
    try:
        daemon.run()
    except KeyboardInterrupt:
        print("Caught KeyboardInterrupt, shutting down.")
    finally:
        daemon._running = False
        daemon._toggle_camera_radio("down")

