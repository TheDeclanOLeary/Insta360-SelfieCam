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
WIFI_INTERFACE = "wlx9cefd5f64c06"
ROBOT_IP = "172.16.0.11"
CAMERA_ID_HEX = [0x31, 0x33, 0x35, 0x56, 0x59, 0x44]
GMAIL_USERNAME = "test@example.com"

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

# --- RTMP STREAM STATE & LOGIC ---
rtmp_client = Client()
stream_active = False
frame_queue = asyncio.Queue(maxsize=5)
codec = None

@rtmp_client.on_video_stream(wait=True)
async def process_live_frame(**kwargs):
    """Parses raw H.264 packets, decodes to OpenCV, applies the math crop, and queues JPEGs."""
    global stream_active, codec
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
                
                # Apply live spatial cropping
                h, w, _ = img_matrix.shape
                lens_w = w // 2
                front_lens = img_matrix[:, 0:lens_w]
                cx, cy = lens_w // 2, h // 2
                crop_w = int(lens_w * 0.60)
                crop_h = int(crop_w * 0.75)
                y1, y2 = cy - (crop_h // 2), cy + (crop_h // 2)
                x1, x2 = cx - (crop_w // 2), cx + (crop_w // 2)

                cropped_frame = front_lens[y1:y2, x1:x2]
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
                ret, jpeg = cv2.imencode('.jpg', cropped_frame, encode_param)
                if ret:
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait() # Drop oldest frame to maintain low latency
                        except asyncio.QueueEmpty:
                            pass
                    frame_queue.put_nowait(jpeg.tobytes())
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
    
    # Flush the frame queue
    while not frame_queue.empty():
        try:
            frame_queue.get_nowait()
        except asyncio.QueueEmpty:
            break

async def video_generator():
    """Consumes the frame queue and yields MJPEG HTTP boundaries."""
    frame_delay=1.0/15
    while stream_active:
        try:
            frame_bytes = await asyncio.wait_for(frame_queue.get(), timeout=1.0)
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
        # --- BLE wake disabled for now ---
        # camera_awake_event = threading.Event()
        # ble_thread = threading.Thread(
        #     target=wake_camera_pulse,
        #     args=(camera_awake_event,)
        # )
        # ble_thread.start()

        try:
            update_status("connecting", "Waking up robot Wi-Fi dongle...", 5)
            subprocess.run(
                ["nmcli", "device", "disconnect", WIFI_INTERFACE],
                capture_output=True,
                timeout=10,
            )
            time.sleep(0.5)  # let the interface settle out of its transition

            ssid_found = False
            for i in range(5):
                # Progress scales during Wi-Fi scan (45% to 70%)
                scan_prog = min(45 + 2*i, 65)
                update_status("connecting", f"Scanning airwaves for {CAMERA_SSID} (Attempt {i+1}/5)...", scan_prog)

                # nmcli rate-limits rescans; if it's deferred the cached list is
                # stale, so skip this round rather than trust a phantom SSID.
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
                    # camera_awake_event.set()  # BLE wake disabled for now
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
            
            # Ping the camera's API for up to 10 seconds to ensure its web server is awake
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
            # --- BLE wake disabled for now ---
            # camera_awake_event.set()
            # ble_thread.join(timeout=8)
            # clear_adv_instances()
            pass

    elif state == "down":
        update_status("idle", "Tearing down camera connection...", 0)
        # clear_adv_instances()  # BLE wake disabled for now
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
async def stream_feed():
    """Initializes the RTMP connection and serves the MJPEG feed."""
    try:
        # Offload the blocking network connection to a thread
        await asyncio.to_thread(start_rtmp_stream)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start live stream: {e}")
    
    return StreamingResponse(video_generator(), media_type="multipart/x-mixed-replace; boundary=frame")
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
    # Move the arm to the selfie pose (follows the trajectory if not already
    # there). The arm connects on first move, not at import.
    try:
        arm_control.move_arm_to_selfie()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Arm positioning failed: {e}")


def _capture_to_file(unique_image_path):
    """Drive one camera capture over HTTP and save the result to disk.
    Raises requests connection errors up to the caller so they can auto-reset."""
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
    # CRITICAL: Stop the RTMP feed to free up network and camera ISP before triggering the capture
    stop_rtmp_stream()
    time.sleep(1.0) # Hardware settling delay

    unique_image_path = "./static/latest.jpg"

    try:
        try:
            _capture_to_file(unique_image_path)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            # Camera HTTP dropped ("Max retries exceeded" / network unreachable).
            # Auto-reset the connection over the Wi-Fi/HTTP path only — no BLE,
            # no arm — then retry the capture once.
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
    """Best-effort: there is no direct OSC power-off, so set the standard
    sleepDelay/offDelay options to make the camera sleep and power itself off
    shortly after we disconnect. Must run while Wi-Fi is still up. Errors ignored."""
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
    # CRITICAL: Stop the RTMP feed before issuing network commands to disconnect
    stop_rtmp_stream()

    # The arm only returns home on an explicit Disconnect press (home_arm=true).
    # Automatic teardown — idle timeout, page hidden/closed — tears down the
    # camera but leaves the arm exactly where it is. If homing fails we still
    # tear down the camera so the user isn't left connected, but report it.
    arm_error = None
    if home_arm:
        try:
            arm_control.move_arm_home()
            arm_control.arm_release()
        except Exception as e:
            arm_error = str(e)

    # Tell the camera to power off (best-effort) while it's still reachable,
    # then tear down the Wi-Fi link.
    power_off_camera()
    try:
        toggle_camera_radio("down")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if arm_error:
        return {"status": "warning", "message": f"Camera disconnected; arm home/release issue: {arm_error}"}
    return {"status": "success", "message": "Camera disconnected."}
