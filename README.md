🐍 Nāga – ML-Based Snake Identification System

📌 Overview

Nāga is an AI-powered system designed to validate and identify snake images using a multi-stage machine learning pipeline. The system analyzes uploaded images to determine whether the image contains a snake, checks its validity, and predicts the snake species using deep learning models.

This project integrates Computer Vision, API development, and Conversational AI to provide a reliable and user-friendly solution for real-world applications.


🚀 Features
 • 🐍 Snake vs Non-Snake Detection
 • ✅ Image Validity Checking (valid / invalid)
 • ⚠️ Invalid Reason Classification (blur, low light, etc.)
 • 🔍 Snake Species Identification using CNN (MobileNetV2)
 • 💬 Chatbot Support using Rasa
 • 🌐 Web-based User Interface


🧠 Tech Stack

🔹 Machine Learning
 • TensorFlow / Keras
 • CNN (MobileNetV2)
 • Image preprocessing & augmentation

🔹 Backend
 • FastAPI (REST API development)
 • ngrok (for API exposure during development)

🔹 Frontend
 • HTML, CSS, JavaScript

🔹 Chatbot
 • Rasa Framework (with AI fallback)

🔹 Tools
 • Git & GitHub
 • Postman (API testing)


⚙️ System Architecture

Frontend (HTML/CSS/JS)
        ↓
     FastAPI
        ↓
 ML Models (Validation + Identification)
        ↓
     Rasa Chatbot



📂 Project Structure

Nāga/
│── frontend/              # HTML, CSS, JS files
│── backend/               # FastAPI backend
│── models/                # Trained ML models
│── chatbot/               # Rasa chatbot files
│── dataset/               # Image dataset (optional/not included)
│── README.md



▶️ How to Run the Project

1️⃣ Clone the Repository

git clone https://github.com/zaviruuu/Naga--ML-Based-Snake-Identifier-for-Sri-Lanka-.git
cd Naga--ML-Based-Snake-Identifier-for-Sri-Lanka-



2️⃣ Setup Backend (FastAPI)

cd backend
pip install -r requirements.txt
uvicorn main:app --reload



3️⃣ Run Chatbot (Rasa)

cd chatbot
rasa run actions
rasa run



4️⃣ Connect using ngrok (optional)

ngrok http 8000



5️⃣ Open Frontend
 • Open index.html in browser
 • Upload image → Get results


📊 Model Details
 • Architecture: MobileNetV2 (Transfer Learning)
 • Task:
 • Snake Detection
 • Image Validation
 • Species Identification
 • Evaluation Metrics:
 • Accuracy
 • Precision
 • Recall


⚠️ Limitations
 • Performance depends on image quality
 • Limited dataset may affect generalization
 • No real-time video detection


🔮 Future Improvements
 • Mobile app development
 • Real-time snake detection
 • Larger dataset for improved accuracy
 • Advanced chatbot with better NLP


👨‍💻 Team Members
 • Saviru Wickramarathne
 • Savithmee Weerasinghe
 • Ketheeswaran Januran


🙏 Acknowledgment

We would like to express our sincere gratitude to Dr. Ruvan Weerasinghe for his guidance, support, and valuable feedback throughout the project.


📜 License

This project is developed for academic purposes.
