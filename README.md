[project_readme.md](https://github.com/user-attachments/files/34678351/project_readme.md)
# Crowd Monitoring & Incident Response System

A real-time crowd monitoring and incident management application built with FastAPI and YOLO (You Only Look Once) object detection. This system provides live metrics, crowd density analysis, and an automated incident state machine, all viewable through a live WebSocket-powered dashboard.

## Features

* **Real-Time Crowd Detection:** Utilizes a YOLO computer vision model to detect and count people in video feeds.
* **Live Dashboard:** An interactive HTML frontend (`index_2.html`) that displays live metrics and crowd data using WebSockets for low-latency updates.
* **Incident State Machine:** Automated tracking and management of incidents (e.g., normal, alert, critical) handled by a dedicated module (`incidents_2.py`).
* **High Performance:** Built on FastAPI for rapid API responses and efficient asynchronous WebSocket connections.

## Project Structure

* `app_2.py`: The main FastAPI application entry point. Handles API routing, WebSocket connections, and integrates the YOLO model.
* `incidents_2.py`: Contains the logic and state machine for evaluating crowd metrics and triggering/managing incident states.
* `index_2.html`: The frontend dashboard interface that connects to the backend to display real-time video streams and statistics.
* `requirements_2.txt`: Lists all Python dependencies required to run the project.
* `*.pt` / YOLO weights: The binary model file used for object detection (ensure this is placed in the correct directory as referenced in `app_2.py`).

## Prerequisites

* Python 3.8+
* A working camera or video feed (depending on how `app_2.py` is configured).

## Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <your-repository-url>
   cd <your-project-directory>
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements_2.txt
   ```

## Usage

1. Start the FastAPI server using Uvicorn:
   ```bash
   uvicorn app_2:app --reload
   ```
   *(Note: If your FastAPI instance inside `app_2.py` is named differently than `app`, adjust the command accordingly).*

2. Open your web browser and navigate to the local server address provided by Uvicorn (usually `http://127.0.0.1:8000`). 
3. The `index_2.html` dashboard should load and establish a WebSocket connection to begin displaying live data.

## Technologies Used

* **Backend:** [FastAPI](https://fastapi.tiangolo.com/), Python
* **Computer Vision:** YOLO (Ultralytics), OpenCV
* **Frontend:** HTML5, JavaScript, WebSockets
