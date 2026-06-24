# S-Bot Selfie Station

A FastAPI application that operates an automated selfie station using a robotic arm and an Insta360 camera. This project provides a full web interface to deploy the arm, connect to the camera, stream a live video feed, capture a photo, and email the result. It features branding for the University of Tulsa Institute for Robotics & Autonomy and L5vel.

---

## Features

* **Hardware Automation:** Automatically connects to the Insta360 X5 via Wi-Fi using `nmcli` and routes the subnet.
* **Live Preview:** Decodes an RTMP live stream using `av` and serves an MJPEG feed to the browser.
* **Image Processing:** Applies real-time spatial cropping using OpenCV to isolate the front lens view.
* **Arm Coordination:** Controls an xArm over the network to move into a designated selfie pose before capturing the photo, and safely returns it home on disconnect.
* **Delivery:** Integrates with SMTP to send the captured image directly to a user-provided email address.

---

## Prerequisites

### Hardware
* **Camera:** Insta360 X5 (Default SSID: `X5 135VYD.OSC`, IP: `192.168.42.1`).
* **Robot Arm:** xArm configured at IP `172.16.0.11`.
* **Wi-Fi Dongle:** A dedicated wireless interface (configured in the code as `wlx9cefd5f64c06`).

### System Packages
This application relies on Linux networking and Bluetooth utilities for hardware handshakes:
* `NetworkManager` (specifically `nmcli`).
* `bluez` (specifically `btmgmt`).

---

## Environment Setup

You must configure an environment variable with a Gmail App Password for the SMTP email functionality to work.

```bash
# Export the required email password
export GMAIL_APP_PASSWORD="your_app_password"
```

---

## Running the Application

1.  Ensure your virtual environment is active and all dependencies (e.g., `fastapi`, `opencv-python`, `av`, `requests`) are installed.
2.  Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3.  Access the interface at `http://localhost:8000`.

---

## Project Structure

* **`main.py`**: The FastAPI backend handling camera networking, RTMP streaming, OpenCV processing, and SMTP email delivery.
* **`arm_control.py`**: The hardware abstraction layer managing the robotic arm's movement trajectories and deployment state.
* **`index.html`**: The frontend user interface featuring connection progress tracking, live video rendering, and photo download/email controls.
* **`static/`**: Directory for storing frontend assets and temporarily saving captured photos (e.g., `latest.jpg`).
