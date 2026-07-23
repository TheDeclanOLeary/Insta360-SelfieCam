import os
import shutil
import smtplib
import subprocess
import time
import uuid
import sys
import threading
import asyncio
from contextlib import asynccontextmanager
from email.message import EmailMessage
from pathlib import Path

import cv2
import requests
import numpy as np
import av
import arm_control
from insta360.rtmp import Client
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles


# --- CONFIGURATION ---
CAMERA_SSID = "X5 135VYD.OSC"
CAMERA_IP = "192.168.42.1"
WIFI_PROFILE_NAME = "Insta360"
WIFI_INTERFACE = "wlx9cefd5f89420"
ROBOT_IP = "172.16.0.11"
CAMERA_ID_HEX = [0x31, 0x33, 0x35, 0x56, 0x59, 0x44]
GMAIL_USERNAME = "olearyd74@gmail.com"

# Global lock for video streaming
stream_lock = threading.Lock()

# Get environment variable
GMAIL_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")

if not GMAIL_PASSWORD:
    raise KeyError(
        "Environment variable 'GMAIL_APP_PASSWORD' is not set. "
        "Please set it in your shell before running this script."
    )

# --- GLOBAL STATUS TRACKER ---
connection_state = {
    "status": "idle", # idle, connecting, connected, error
    "message": "Waiting to connect...",
    "progress": 0
}

def update_status(status: str, message: str, progress: int):
    """Updates the global state and mirrors it to the terminal."""
    connection_state["status"] = status
    connection_state["message"] = message
    connection_state["progress"] = progress
    print(f"[{progress}%] {message}")


# Initialize FastAPI
app = FastAPI()

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# --- EQUIRECTANGULAR MAPPING LOGIC ---
map_x, map_y = None, None

import numpy as np

def generate_equirectangular_maps(w, h, config=None):
    """
    Production-ready LUT Generator for Dual-Fisheye to Equirectangular conversion.
    Includes tuning parameters to fix seam alignment, FOV scaling, and optical centers.
    """
    if config is None:
        config = {
            "fov_deg": 194,         # Typical dual-fisheye lenses have > 180 deg FOV
            "yaw_offset_deg": 0.0,    # Overall horizontal rotation
            "cx1_offset": 0.0,        # Front lens X center offset
            "cy1_offset": 0.0,        # Front lens Y center offset
            "cx2_offset": 0.0,        # Rear lens X center offset
            "cy2_offset": 0.0,        # Rear lens Y center offset
            "radius_scale": 1.0       # Fine-tune radius scaling
        }

    out_w, out_h = w, w // 2
    u, v = np.meshgrid(np.linspace(0, 1, out_w), np.linspace(0, 1, out_h))

    yaw_offset_rad = np.radians(config["yaw_offset_deg"])
    max_theta = np.radians(config["fov_deg"]) / 2.0

    # Longitude (theta) and Latitude (phi)
    theta = (u - 0.5) * 2 * np.pi + yaw_offset_rad
    phi = (0.5 - v) * np.pi

    # 3D Cartesian coordinates on a unit sphere
    x = np.cos(phi) * np.cos(theta)  # Forward/Backward
    y = np.cos(phi) * np.sin(theta)  # Left/Right
    z = np.sin(phi)                  # Up/Down

    # Lens centers with configurable offsets
    lens_radius = (w / 4) * config["radius_scale"]
    cx1 = (w / 4) + config["cx1_offset"]
    cy1 = (h / 2) + config["cy1_offset"]
    cx2 = (3 * w / 4) + config["cx2_offset"]
    cy2 = (h / 2) + config["cy2_offset"]

    front_mask = x >= 0

    fisheye_x = np.zeros_like(u, dtype=np.float32)
    fisheye_y = np.zeros_like(v, dtype=np.float32)

    # --- FRONT LENS MAPPING (x >= 0) ---
    theta_front = np.arccos(np.clip(x[front_mask], -1.0, 1.0))
    r_front = lens_radius * (theta_front / max_theta)
    alpha_front = np.arctan2(z[front_mask], y[front_mask])
    
    fisheye_x[front_mask] = cx1 + r_front * np.cos(alpha_front)
    fisheye_y[front_mask] = cy1 - r_front * np.sin(alpha_front)

    # --- REAR LENS MAPPING (x < 0) ---
    theta_rear = np.arccos(np.clip(-x[~front_mask], -1.0, 1.0))
    r_rear = lens_radius * (theta_rear / max_theta)
    alpha_rear = np.arctan2(z[~front_mask], -y[~front_mask])
    
    fisheye_x[~front_mask] = cx2 + r_rear * np.cos(alpha_rear)
    fisheye_y[~front_mask] = cy2 - r_rear * np.sin(alpha_rear)

    return fisheye_x, fisheye_y


# --- RTMP STREAM STATE & LOGIC ---
rtmp_client = Client()
stream_active = False

# Isolate queues so consuming one endpoint doesn't break the other
frame_queue_cropped = asyncio.Queue(maxsize=5)
frame_queue_equirec = asyncio.Queue(maxsize=5)
codec = None

@rtmp_client.on_video_stream(wait=True)
async def process_live_frame(**kwargs):
    """Parses raw H.264 packets, distributes frames to cropped and equirectangular queues."""
    global stream_active, codec, map_x, map_y
    if not stream_active or codec is None:
        return

    content = kwargs.get('content') or kwargs.get('data') or kwargs.get('payload') or kwargs.get('buffer')
    if not content:
        return

    try:
        packets = codec.parse(content)
        for packet in packets:
            frames = codec.decode(packet)
            for frame in frames:
                if not stream_active:
                    return
                
                img_matrix = frame.to_ndarray(format='bgr24')
                h, w, _ = img_matrix.shape
                
                # --- 1. CROPPED FEED ---
                lens_w = w // 2
                front_lens = img_matrix[:, 0:lens_w]
                cx, cy = lens_w // 2, h // 2
                crop_w = int(lens_w * 0.60)
                crop_h = int(crop_w * 0.75)
                y1, y2 = cy - (crop_h // 2), cy + (crop_h // 2)
                x1, x2 = cx - (crop_w // 2), cx + (crop_w // 2)

                cropped_frame = front_lens[y1:y2, x1:x2]
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                ret1, jpeg1 = cv2.imencode('.jpg', cropped_frame, encode_param)
                if ret1:
                    if frame_queue_cropped.full():
                        try:
                            frame_queue_cropped.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    frame_queue_cropped.put_nowait(jpeg1.tobytes())

                # --- 2. EQUIRECTANGULAR FEED ---
                # Initialize transformation matrices on the first frame
                if map_x is None or map_y is None or map_x.shape[1] != w:
                    map_x, map_y = generate_equirectangular_maps(w, h)

                equi_frame = cv2.remap(img_matrix, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                ret2, jpeg2 = cv2.imencode('.jpg', equi_frame, encode_param)
                if ret2:
                    if frame_queue_equirec.full():
                        try:
                            frame_queue_equirec.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    frame_queue_equirec.put_nowait(jpeg2.tobytes())

    except Exception as e:
        print(f"RTMP Decode Error: {e}")

def start_rtmp_stream():
    global stream_active, codec
    if stream_active:
        return
    # Re-initialize the FFmpeg codec context for a fresh stream
    codec = av.CodecContext.create('h264', 'r')
    rtmp_client.open()
    rtmp_client.start_preview_stream()
    stream_active = True

def stop_rtmp_stream():
    global stream_active
    stream_active = False
    try:
        rtmp_client.close()
    except Exception:
        pass
    
    # Flush both frame queues
    for q in (frame_queue_cropped, frame_queue_equirec):
        while not q.empty():
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                break

async def video_generator(queue: asyncio.Queue):
    """Consumes a target frame queue and yields MJPEG HTTP boundaries."""
    frame_delay = 1.0 / 15
    while stream_active:
        try:
            frame_bytes = await asyncio.wait_for(queue.get(), timeout=1.0)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            await asyncio.sleep(frame_delay)
        except asyncio.TimeoutError:
            continue
        except Exception:
            break


# --- BLUETOOTH WAKE LOGIC ---

def run_btmgmt_safe(cmd_list, timeout_sec=3):
    full_cmd = ["sudo", "/usr/bin/timeout", str(timeout_sec), "/usr/bin/btmgmt"] + cmd_list
    return subprocess.run(full_cmd, capture_output=True, text=True)

def clear_adv_instances():
    for i in range(1, 4):
        run_btmgmt_safe(["rm-adv", str(i)], timeout_sec=2)

def wake_camera_pulse(stop_event: threading.Event):
    manuf_data = [
        0x4C, 0x00, 0x02, 0x15, 0x09, 0x4F, 0x52, 0x42, 
        0x49, 0x54, 0x09, 0xFF, 0x0F, 0x00, 
        *CAMERA_ID_HEX, 
        0x00, 0x00, 0x00, 0x00, 0xE4, 0x01,
    ]

    ad_element = [len(manuf_data) + 1, 0xFF] + manuf_data
    hex_payload = "".join(f"{b:02x}" for b in ad_element)

    update_status("connecting", "Purging orphaned Bluetooth states...", 5)
    clear_adv_instances()

    for attempt in range(1, 15):
        if stop_event.is_set():
            break

        # Progress scales slowly during the BLE phase (10% to 40%)
        current_progress = min(10 + (attempt * 2), 40)
        update_status("connecting", f"Transmitting BLE wake pulse {attempt}/14...", current_progress)
        
        result = run_btmgmt_safe(["add-adv", "-c", "-p", "-d", hex_payload, "1"], timeout_sec=3)
        if result.returncode != 0 and result.returncode != 124:
            print(f"     [!] Pulse warning: {result.stderr.strip()}")

        interrupted = stop_event.wait(1.5)
        run_btmgmt_safe(["rm-adv", "1"], timeout_sec=2)
        
        if interrupted:
            update_status("connecting", "Camera radio detected! Halting BLE pulses.", 45)
            break
            
        time.sleep(0.2)


# --- NETWORK & CAMERA LOGIC ---

def toggle_camera_radio(state: str):
    if state == "up":
        update_status("connecting", "Initializing wake sequence...", 0)

        try:
            update_status("connecting", "Waking up robot Wi-Fi dongle...", 5)
            subprocess.run(
                ["nmcli", "device", "disconnect", WIFI_INTERFACE],
                capture_output=True,
                timeout=10,
            )
            time.sleep(0.5)  # let the interface settle out of its transition

            ssid_found = False
            for i in range(10):
                scan_prog = min(45 + 2*i, 65)
                update_status("connecting", f"Scanning airwaves for {CAMERA_SSID} (Attempt {i+1}/10)...", scan_prog)

                rescan = subprocess.run(
                    ["nmcli", "device", "wifi", "rescan"], capture_output=True, text=True
                )
                if rescan.returncode != 0:
                    print(f"     [!] Rescan deferred: {rescan.stderr.strip()}")
                    time.sleep(2.0)
                    continue

                time.sleep(1.5)  # let fresh results populate before reading the list
                scan_results = subprocess.run(
                    ["nmcli", "-t", "-f", "SSID", "device", "wifi", "list"],
                    capture_output=True,
                    text=True,
                )

                if CAMERA_SSID in scan_results.stdout:
                    ssid_found = True
                    update_status("connecting", "Camera AP found! Securing connection...", 75)
                    break

                time.sleep(1.0)

            if not ssid_found:
                update_status("error", "Scan Timeout: Ensure the camera is powered on and broadcasting its Wi-Fi network.", 0)
                raise Exception("Scan Timeout: Ensure the camera is powered on and broadcasting its Wi-Fi network.")

            cmd = ["nmcli", "--wait", "10", "connection", "up", WIFI_PROFILE_NAME]
            result = None
            for attempt in range(2):
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
                if result.returncode == 0:
                    break
                print(f"     [!] Handshake attempt {attempt + 1}/2 failed: {result.stderr.strip()}")
                time.sleep(1.0)

            if result.returncode != 0:
                reason = result.stderr.strip() or "unknown nmcli error"
                update_status("error", f"Wi-Fi Handshake Failed: Try rebooting the camera", 0)
                raise Exception(f"Wi-Fi handshake failed: Try rebooting the camera")

            update_status("connecting", "Wi-Fi authenticated. Waiting for DHCP lease...", 85)
            dhcp_settled = False

            for i in range(6):
                route_check = subprocess.run(
                    ["ip", "route", "show", "dev", WIFI_INTERFACE],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )

                if "192.168.42.0" in route_check.stdout:
                    dhcp_settled = True
                    update_status("connecting", "DHCP assigned! Subnet routing initialized.", 95)
                    break
                time.sleep(0.5)

            if not dhcp_settled:
                subprocess.run(
                    ["nmcli", "connection", "down", WIFI_PROFILE_NAME],
                    capture_output=True,
                    timeout=10,
                )
                update_status("error", "DHCP Timeout.", 0)
                raise Exception("DHCP Timeout: Connected to Wi-Fi, but failed to obtain a local IP address.")

            update_status("connecting", "Wi-Fi routed. Verifying camera API readiness...", 98)
            api_ready = False
            
            for _ in range(10):
                try:
                    res = requests.get(f"http://{CAMERA_IP}/osc/info", timeout=1)
                    if res.status_code == 200:
                        api_ready = True
                        break
                except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
                    pass
                time.sleep(1.0)

            if not api_ready:
                subprocess.run(["nmcli", "connection", "down", WIFI_PROFILE_NAME], capture_output=True)
                update_status("error", "Camera Wi-Fi connected, but API failed to boot.", 0)
                raise Exception("Camera Wi-Fi connected, but the internal web server failed to respond.")

            time.sleep(0.5)
            update_status("connected", "Camera successfully connected and ready.", 100)
        finally:
            pass

    elif state == "down":
        update_status("idle", "Tearing down camera connection...", 0)
        subprocess.run(
            ["nmcli", "connection", "down", WIFI_PROFILE_NAME],
            capture_output=True,
            timeout=10,
        )
        update_status("idle", "Camera disconnected.", 0)
        

def crop_front_lens(image_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image at {image_path} for cropping.")

    h, w, _ = img.shape
    lens_w = w // 2
    front_lens = img[:, 0:lens_w]
    cx, cy = lens_w // 2, h // 2
    crop_w = int(lens_w * 0.60)
    crop_h = int(crop_w * 0.75)
    y1, y2 = cy - (crop_h // 2), cy + (crop_h // 2)
    x1, x2 = cx - (crop_w // 2), cx + (crop_w // 2)
    cv2.imwrite(image_path, front_lens[y1:y2, x1:x2])

def apply_border_and_logo(
    image_path: str, 
    logo_path: str, 
    border_width: int = 200, 
    border_color: tuple = (0, 53, 149)
) -> None:
    """Applies a padded border and bottom-right logo using true alpha blending."""
    img = cv2.imread(image_path)
    if img is None:
        raise Exception(f"Could not read image at {image_path} for border overlay.")

    # Convert RGB tuple to OpenCV BGR
    bgr_border_color = (border_color[2], border_color[1], border_color[0])
    
    # Pad the canvas
    canvas = cv2.copyMakeBorder(
        img, 0, border_width, 0, 0, 
        cv2.BORDER_CONSTANT, value=bgr_border_color
    )
    
    if os.path.exists(logo_path):
        logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)
        
        # Ensure it loaded and has an alpha channel (BGRA)
        if logo is not None and logo.shape[2] == 4:
            # Scale logo to fit border
            max_logo_height = int(border_width * 0.8)
            if logo.shape[0] > max_logo_height:
                scale_factor = max_logo_height / logo.shape[0]
                new_width = int(logo.shape[1] * scale_factor)
                logo = cv2.resize(logo, (new_width, max_logo_height), interpolation=cv2.INTER_AREA)
            
            # Separate the BGR channels from the Alpha channel
            logo_bgr = logo[:, :, 0:3]
            alpha_channel = logo[:, :, 3]
            
            # Normalize the alpha channel to a 0.0 - 1.0 range and broadcast to 3 channels
            alpha_factor = alpha_channel.astype(float) / 255.0
            alpha_factor = np.dstack([alpha_factor, alpha_factor, alpha_factor])
            
            logo_h, logo_w, _ = logo_bgr.shape
            canvas_h, canvas_w, _ = canvas.shape
            
            # Position at bottom right
            margin = int(border_width * 0.1)
            y1 = canvas_h - logo_h - margin
            y2 = canvas_h - margin
            x1 = canvas_w - logo_w - margin
            x2 = canvas_w - margin
            
            # Extract the Region of Interest (ROI) from the background
            roi = canvas[y1:y2, x1:x2].astype(float)
            logo_fg = logo_bgr.astype(float)
            
            # Perform true alpha blending
            blended = cv2.multiply(logo_fg, alpha_factor) + cv2.multiply(roi, 1.0 - alpha_factor)
            
            # Paste the blended image back into the canvas
            canvas[y1:y2, x1:x2] = blended.astype(np.uint8)

    # Overwrite the temporary file
    cv2.imwrite(image_path, canvas)
def send_email_attachment(recipient: str, filepath: str):
    sender = GMAIL_USERNAME
    app_password = GMAIL_PASSWORD
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = "Your Selfie!", sender, recipient
    msg.set_content("Here is your photo!")
    try:
        with open(filepath, "rb") as f:
            msg.add_attachment(
                f.read(), maintype="image", subtype="jpeg", filename="selfie.jpg"
            )
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as server:
            server.login(sender, app_password)
            server.send_message(msg)
    except Exception as e:
        raise Exception(f"Email failed: {e}")


# --- API ENDPOINTS ---

@app.get("/")
async def serve_frontend():
    return FileResponse("index.html")

@app.get("/stream")
async def stream_feed_cropped():
    """Initializes the RTMP connection and serves the cropped MJPEG feed."""
    try:
        await asyncio.to_thread(start_rtmp_stream)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start live stream: {e}")
    
    return StreamingResponse(video_generator(frame_queue_cropped), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stream/equirec")
async def stream_feed_equirec():
    """Initializes the RTMP connection and serves the full equirectangular MJPEG feed."""
    try:
        await asyncio.to_thread(start_rtmp_stream)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start live stream: {e}")
    
    return StreamingResponse(video_generator(frame_queue_equirec), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status")
async def get_status():
    """Non-blocking endpoint so the frontend can poll the connection state."""
    return connection_state

@app.post("/connect")
def connect_camera():
    try:
        toggle_camera_radio("up")
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/position-arm")
def position_arm():
    try:
        arm_control.move_arm_to_selfie()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arm positioning failed: {e}")


def _capture_to_file(unique_image_path):
    execute_url = f"http://{CAMERA_IP}/osc/commands/execute"
    trigger_res = requests.post(
        execute_url, json={"name": "camera.takePicture"}, timeout=5
    ).json()

    if trigger_res.get("state") == "error":
        raise Exception(
            "Camera rejected capture command. Make sure it's in Photo mode and then refresh the page"
        )

    command_id = trigger_res.get("id")
    status_url = f"http://{CAMERA_IP}/osc/commands/status"
    file_url = None

    for _ in range(10):
        time.sleep(1)
        status_res = requests.post(
            status_url, json={"id": command_id}, timeout=5
        ).json()
        if status_res.get("state") == "done":
            file_url = status_res.get("results", {}).get("fileUrl")
            break

    if not file_url:
        raise Exception("Camera timed out while processing image.")

    img_data = requests.get(file_url, timeout=60).content

    with open(unique_image_path, "wb") as f:
        f.write(img_data)


@app.post("/capture")
def capture_image():

    stop_rtmp_stream()
    time.sleep(1.0) 

    unique_image_path = "./static/latest.jpg"
    logo_path = "./static/logo.png"

    try:
        try:
            _capture_to_file(unique_image_path)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            update_status("connecting", "Camera connection lost. Auto-resetting...", 0)
            toggle_camera_radio("up")
            _capture_to_file(unique_image_path)

    except Exception as e:
        try:
            toggle_camera_radio("down")
        except:
            pass
        raise HTTPException(status_code=500, detail=str(e))

    try:
        crop_front_lens(unique_image_path)
        apply_border_and_logo(unique_image_path, logo_path)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed during local image processing: {e}"
        )

    return {"status": "success", "lan_path": f"/static/latest.jpg?t={int(time.time())}"}

@app.post("/email")
def trigger_email(email: str, image: str):
    clean_path = image.split("?")[0].lstrip("/")

    if not os.path.exists(clean_path):
        raise HTTPException(
            status_code=404, detail="Requested image file not found on server."
        )

    try:
        send_email_attachment(email, clean_path)
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def power_off_camera():
    try:
        execute_url = f"http://{CAMERA_IP}/osc/commands/execute"
        res = requests.post(
            execute_url,
            json={"name": "camera.setOptions",
                  "parameters": {"options": {"sleepDelay": 15, "offDelay": 30}}},
            timeout=5,
        ).json()
        if res.get("state") == "error":
            print(f"     [!] Camera rejected power-off options: {res.get('error')}")
    except Exception as e:
        print(f"     [!] Camera power-off request failed: {e}")


@app.post("/disconnect")
def disconnect_camera(home_arm: bool = False):
    stop_rtmp_stream()

    arm_error = None
    if home_arm:
        try:
            arm_control.move_arm_home()
            arm_control.arm_release()
        except Exception as e:
            arm_error = str(e)

    power_off_camera()
    try:
        toggle_camera_radio("down")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if arm_error:
        return {"status": "warning", "message": f"Camera disconnected; arm home/release issue: {arm_error}"}
    return {"status": "success", "message": "Camera disconnected."}