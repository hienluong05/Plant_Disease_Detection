# 🌿 Plant Disease Detection

A deep learning-based application for detecting plant leaf diseases using PyTorch and a simple web demo built with Flask. Users can upload an image of a leaf, and the system will predict the corresponding disease.

---

## 📁 Project Structure

```
MyAI/
├── App/                         # Virtual environment and configurations
├── flaskr/                      # Flask web application
│   ├── static/                  # Static files (CSS/JS/images if any)
│   ├── templates/
│   │   └── index.html           # Web interface
│   ├── __init__.py              # Initialize Flask app
│   ├── ai.py                    # Model loading and prediction logic
│   ├── home.py                  # Flask routes and logic
│   └── plant_disease_detection.pt  # Trained PyTorch model file
├── instance/
├── tests/                       # Test code (optional)
├── model/                       # PyTorch training code
├── PlantVillage/                # PlantVillage dataset used for training
```

---

## 🔧 Technologies Used

- **PyTorch:** For building and training the image classification model.
- **Torchvision:** For preprocessing images (resizing, normalization, etc.).
- **Flask:** For creating a lightweight web interface.
- **HTML/CSS:** For the frontend of the web demo.
- **PlantVillage Dataset:** Public dataset for training the plant disease classifier.

---

## 🚀 How to Run the Web Demo

1. **Set up the environment:**
   - Ensure you have Python 3.8 installed.
   - Create and activate a Python Virtual Environment ([docs](https://docs.python.org/3/tutorial/venv.html))
   - Install required packages:
     ```bash
     pip install flask torch torchvision pandas numpy
     ```

2. **Run the Flask app:**
   - Clone the repository, move to the `flaskr` folder and run:
     ```bash
     flask --app flaskr run
     ```
   - Open your browser and go to:  
     [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🧠 How It Works

1. The user uploads a leaf image via the web interface.
2. The image is preprocessed using `torchvision.transforms`.
3. The trained PyTorch model (`.pt` file) predicts the class of the disease.
4. The prediction result is displayed on the web page.

---

## 📝 Dataset

We used the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), which contains over 50,000 images of healthy and diseased plant leaves categorized into 38 classes.

---

## 📌 Notes

- The trained model file `plant_disease_detection.pt` **must be placed inside the `flaskr/` folder**.
- You can retrain the model using the code in the `model/` directory with your own parameters or architectures.

---

## 📌 Contact me
For questions, suggestions, or contributions, please open an issue or contact: ms.dangthihienluong@gmail.com
