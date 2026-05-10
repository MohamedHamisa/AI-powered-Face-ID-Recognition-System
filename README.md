# Face ID Recognition System

🚀 AI-powered Face Recognition System built during the **DEBI AI, Data Science & Cloud Computing Hackathon** using FaceNet, MTCNN, PyTorch, OpenCV, and Streamlit.

The system can:
- Register faces into a database
- Generate 512D embeddings
- Recognize people in real-time
- Compare faces using cosine similarity
- Work with images, videos, or live camera feeds

---

# 🏆 Hackathon Project

This project was developed as part of the:

## DEBI AI, Data Science & Cloud Computing Hackathon

The challenge focused on building real-world AI solutions using modern machine learning and cloud technologies.

Our solution was designed to demonstrate:
- Deep Learning
- Computer Vision
- Real-time AI inference
- Modular software architecture
- Production-ready deployment workflow

---

# ✨ Features

✅ Real-time face recognition  
✅ Face registration system  
✅ Multi-face detection  
✅ 512D FaceNet embeddings  
✅ Cosine similarity matching  
✅ Streamlit interactive UI  
✅ Google Drive backup support  
✅ GPU acceleration with CUDA  
✅ Lightweight JSON database  

---

# 🧠 How It Works

Instead of comparing raw images directly, the system converts every detected face into a unique:

## 512-Dimensional Embedding Vector

These embeddings are generated using:

- FaceNet (InceptionResnetV1)
- MTCNN face alignment

The recognition engine then compares embeddings using:

## Cosine Similarity

This allows robust recognition even when:
- Lighting changes
- Facial expressions change
- The user wears glasses
- Camera angles vary slightly

---

# 🏗️ Project Architecture

```text
app.py
│
├── recognition.py
│     ├── model.py
│     └── database.py
```

---

# 📂 Components

## model.py
Responsible for:
- Face detection
- Face alignment
- Embedding generation
- Deep learning inference

Uses:
- MTCNN
- FaceNet (InceptionResnetV1)

---

## database.py
Responsible for:
- Saving embeddings
- Loading registered users
- Managing JSON storage

---

## recognition.py
Responsible for:
- Recognition pipeline
- Cosine similarity calculations
- Confidence scoring
- Identity matching

---

## app.py
Responsible for:
- Streamlit UI
- Camera integration
- Registration workflow
- Real-time recognition display

---

# 🔄 Registration Workflow

1. Upload image
2. Detect face using MTCNN
3. Crop and align face
4. Generate 512D embedding
5. Save embedding into database

---

# 🎯 Recognition Workflow

1. Capture frame/image
2. Detect all faces
3. Generate embeddings
4. Compare against database
5. Return closest match
6. Display confidence score

---

# ⚡ Tech Stack

| Technology | Purpose |
|---|---|
| PyTorch | Deep Learning |
| FaceNet | Face embeddings |
| MTCNN | Face detection |
| OpenCV | Image processing |
| Streamlit | Web UI |
| NumPy | Vector operations |
| PIL | Image handling |

---

# 📦 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/face-id-system.git

cd face-id-system
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📋 Requirements

```txt
streamlit>=1.33.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
Pillow>=10.0.0
torch>=2.2.0
torchvision>=0.17.0
facenet-pytorch>=2.5.3
```

---

# ▶️ Run Application

```bash
streamlit run app.py
```

---

# ☁️ Deployment

The project was optimized for:

- Google Colab
- NVIDIA T4 GPU
- Ngrok public hosting
- Google Drive auto-backup

This allows the app to run entirely in the cloud with GPU acceleration.

---

# 🚀 Performance Optimizations

## Improvements Added

- Fixed ambiguous NumPy array comparison bug
- Reduced repeated MTCNN calls
- Faster embedding extraction
- Safer tensor indexing
- Improved multi-face processing
- Better recognition stability

---

# 💼 Real-World Applications

| Domain | Use Case |
|---|---|
| Security | Smart access systems |
| Education | Automatic attendance |
| Healthcare | Patient identity verification |
| Electronics | Face unlock systems |
| Public Events | Entry monitoring |

---

# 🔮 Future Improvements

- Face anti-spoofing
- PostgreSQL/MongoDB support
- Mobile application
- Face tracking
- ONNX/TensorRT acceleration
- Docker deployment
- REST API integration

---

# 👨‍💻 Author

## Mohamed Hamisa

AI & Machine Learning Engineer

Built with:
- Deep Learning
- Computer Vision
- PyTorch
- Face Recognition Technologies

---

# 📜 License

This project is open-source and available for educational and research purposes.
