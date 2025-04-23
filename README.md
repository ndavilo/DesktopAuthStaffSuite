Here's a `README.md` you can use for your **Clock In and Clock Out Facial Recognition System** project:

---

# Face Recognition Attendance System

This project is a **facial recognition-based attendance system** that allows staff to clock in and clock out automatically using their face. It uses **OpenCV**, **InsightFace**, **Redis**, and **cosine similarity** to identify individuals and log their attendance actions.

---

## 🔧 Features

- 🎥 Real-time video capture from webcam
- 🧠 Face detection and recognition using [InsightFace](https://github.com/deepinsight/insightface)
- 🧮 Cosine similarity for facial embedding matching
- 🕒 Automatic action detection (Clock In before 12 PM, Clock Out after 12 PM)
- 🗃️ Attendance logs stored in **Redis**
- 🔁 Prevents duplicate clock-ins or outs within the same day
- 🛠️ Compatible with `pyinstaller` for executable builds

---

## 📦 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```txt
opencv-python-headless
numpy
pandas
redis
insightface
scikit-learn
python-dotenv
```

---

## 📁 Project Structure

```text
.
├── main.py         # Main application file
├── insightface_model/      # Folder containing InsightFace model weights
├── .env                    # Environment variables
├── README.md
└── requirements.txt
```

---

## 🧪 Environment Variables

Create a `.env` file with the following content:

```env
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=
ACTION_INTERVAL=10
DETECTION_THRESHOLD=0.5
DEFAULT_ZONE=Lagos Zone 2
```

---

## 🚀 How It Works

1. On startup, the system loads staff facial embeddings from Redis.
2. Captures live webcam feed and performs facial detection using InsightFace.
3. Extracts facial features (embeddings) and compares them to the database using cosine similarity.
4. If a match is found, the system:
   - Determines if it’s time to Clock In or Clock Out.
   - Checks attendance history to avoid duplicate actions.
   - Saves logs to Redis in the format: `Name@Role@Timestamp@Action`.
5. Displays the result with bounding boxes and text overlays on the live video feed.

---

## 💾 Storing Staff Data in Redis

Facial embeddings should be stored in a Redis hash called `staff:register` with keys in the format:

```txt
FileNo.Name@Role@Zone
```

Each value should be a serialized numpy array (`np.float32`) of facial embeddings.

---

## 🖥️ Running the App

```bash
python clock_in_out.py
```

Press `ESC` to exit the video window.

---

## 📸 GUI vs Headless Mode

The app automatically falls back to **headless mode** (no video display) if OpenCV GUI features are unavailable.

---

## 🛑 Exit Conditions

- Press `ESC` in the video window to stop.
- For headless environments, use `CTRL+C` in terminal.

---

## 🔐 Security Note

This system is for **demonstration/educational purposes**. For production use, consider securing:
- Redis access
- Facial data encryption
- GDPR compliance for biometric data

---

## 📜 License

This project is open-source and free to use under the MIT License.
