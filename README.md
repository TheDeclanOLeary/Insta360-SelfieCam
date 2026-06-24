# S-Bot Selfie Station

A FastAPI application that operates an automated selfie station using a robotic arm and an Insta360 X5 camera. This project provides a full web interface to connect to the camera, stream a live video feed, deploy the arm, capture a photo, and email the result. It features branding for the [University of Tulsa Institute for Robotics & Autonomy](https://utulsa.edu/research/robotics-institute/) and [L5vel](l5vel.com).

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
* **Camera:** Insta360 X5 (Default IP: `192.168.42.1`). The application may work with other Insta360 cameras but has not been tested
* **Robot Arm:** xArm configured at IP `172.16.0.11`. The app can be configured to work with other arms by changing the arm_control file
* **Wi-Fi Dongle:** A *dedicated* wireless interface. If your robot relies on an internet connection, this application will break it unless a second interface is used

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

1.  Ensure your virtual environment is active and all dependencies are installed.
2.  Start the FastAPI server:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

3.  Access the interface at `http://localhost:8000`.

---

## Project Structure

* **`main.py`**: The FastAPI backend handling camera networking, RTMP streaming, OpenCV processing, and SMTP email delivery.
* **`arm_control.py`**: The hardware abstraction layer managing the robotic arm's movement trajectories and deployment state.
* **`index.html`**: The frontend user interface featuring connection controls and live video rendering.
* **`static/`**: Directory for storing frontend assets and temporarily saving captured photos (e.g., `latest.jpg`).
