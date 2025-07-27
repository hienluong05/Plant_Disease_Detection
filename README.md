🌿 Plant Disease Detection
A deep learning-based application for detecting plant leaf diseases using PyTorch and a simple web demo built with Flask. Users can upload an image of a leaf, and the system will predict the corresponding disease.

📁 Project Structure
MyAI/
├── App/                         # Virtual environment and configurations
├── flaskr/                      # Flask web application
│   ├── static/                  # Static files (CSS/JS/images if any)
│   ├── templates/
│   │   └── index.html           # Web interface
│   ├── __init__.py              # Initialize Flask app
│   └── plant_disease_detection.pt  # Trained PyTorch model file
├── instance/
├── tests/                       # Test code (optional)
├── model/                       # PyTorch training code
├── PlantVillage/                # PlantVillage dataset used for training

🔧 Technologies Used
PyTorch: For building and training the image classification model.
Torchvision: For preprocessing images (resizing, normalization, etc.).
Flask: For creating a lightweight web interface.
HTML/CSS: For the frontend of the web demo.
PlantVillage Dataset: Public dataset for training the plant disease classifier.

🚀 How to Run the Web Demo
1. Set up the environment:
   - You must have Python3.8 installed in your machine.
   - Create a Python Virtual Environment & Activate Virtual Environment https://docs.python.org/3/tutorial/venv.html
   - Install Flask, pytorch, pandas, numpy
2. Run the Flask app
   - Clone repository, move to flaskr folder and run command: flask --app flaskr run
   - Then open your browser and go to:
     http://127.0.0.1:5000
 
🧠 How It Works
The user uploads a leaf image via the web interface.
The image is preprocessed using torchvision.transforms.
The trained PyTorch model (.pt file) predicts the class of the disease.
The prediction result is displayed on the web page.

📝 Dataset
We used the PlantVillage dataset, which contains over 50,000 images of healthy and diseased plant leaves categorized into 38 classes.

📌 Notes
The trained model file plant_disease_detection.pt must be placed inside the flaskr/ folder.

You can retrain the model using the code in the model/ directory with your own parameters or architectures.


