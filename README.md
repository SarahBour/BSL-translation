# British Sign Language Translation Web App  
## Description:  
This is a British Sign Language (BSL) translation minimal web app designed to translate sign language gestures into text using a machine learning model. The app is built using Python, Flask, HTML, and CSS for the graphical user interface (GUI).  

I learned BSL before collecting the dataset, ensuring the translation model's training was aligned with real-world usage. The dataset was collected using a webcam to capture sign language gestures, and I have commented out the relevant code cells to prevent the dataset collection from being run again, as the data was already taken and used for training.  

##  Features:  
👐 BSL Gesture Recognition: Recognizes common British Sign Language gestures and translates them into text.  
💻 Web Interface: Simple and clean user interface built with Flask, HTML, and CSS.  
🎥 Webcam Integration: Uses webcam for capturing real-time BSL gestures (the code cells are commented out to avoid redundant dataset collection).  
🧠 Machine Learning Model: Trained on a custom dataset of BSL gestures for accurate translation.  
## Tech Stack:    
🐍 Backend: Python, Flask  
🌐 Frontend: HTML, CSS  
🧠 Machine Learning: Custom dataset, trained using various models in Python  
🎥 Webcam: Used for dataset collection (code commented out to avoid duplicate data capture)  
Getting Started:  
1. **Clone the Repository**:  

```
git clone https://github.com/SarahBour/BSL-translation.git
cd BSL-translation
```

2. **Install the Requirements**:
```
py -m pip install -r requirements.txt
```

3. **Run the Application**:

```
py app.py
```


You can now access the web app locally by visiting http://localhost:5000 in your browser.  

## Important Notes:  
📷 Webcam Dataset Collection:  
The code for webcam dataset collection has been commented out to prevent redundancy. The dataset was collected earlier with careful attention to ensure the quality and accuracy of the model's training.  

💾 Dataset Files:  
Due to file size limitations, the dataset files cannot be uploaded to GitHub. However, if you wish to replicate the training process, the dataset can be acquired separately upon request, or you can use your own dataset to retrain the model.    

🌟If you like this project don't forget to star ⭐ the repository!

