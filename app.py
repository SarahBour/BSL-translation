from flask import Flask, render_template, request, Response, redirect, url_for 
import cv2 
import tensorflow as tf 
import numpy as np 
import mediapipe as mp 
import os 
import pafy 
from vidgear.gears import CamGear


app = Flask(__name__)
model = tf.keras.models.load_model('sign.h5')
mp_draw = mp.solutions.drawing_utils
mp_model = mp.solutions.holistic 
model_lm = mp_model.Holistic(min_detection_confidence = 0.3, min_tracking_confidence = 0.3)
alphabets = np.array(['A', 'B', 'C', 'D', 'E','F','G','H', 'I', 'J', 'K', 'L', 'M', 'N', 'O','P', 'Q','R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z','-'])





app.secret_key = "huhuhusasa992-67tcrrsq3577v"
@app.route("/")
def index():
	return render_template("index.html")



def detecting_landmarks(image):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image.flags.writeable = False 
	results = model_lm.process(image)
	image.flags.writeable = True 
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	return image, results

def draw_connections(image, results):
	mp_draw.draw_landmarks(image, results.right_hand_landmarks, mp_model.HAND_CONNECTIONS)
	mp_draw.draw_landmarks(image, results.left_hand_landmarks, mp_model.HAND_CONNECTIONS)

def get_coordinates(results):
	right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
	left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
	keypoints = np.concatenate([right_hand, left_hand])
	return keypoints 

def translating_web():
	sequence = []
	sentence = []
	threshold = 0.99

	cap = cv2.VideoCapture(0)

	while True:

        # Read feed
		ret, frame = cap.read()

	        # Make detections
		image, results = detecting_landmarks(frame)

	        
	        # Draw landmarks
		draw_connections(image, results)
	        
	        # 2. Prediction logic
		keypoints = get_coordinates(results)
	#         sequence.insert(0,keypoints)
	#         sequence = sequence[:30]
		sequence.append(keypoints)
		sequence = sequence[-10:]
	        
		if len(sequence) == 10:
			res = model.predict(np.expand_dims(sequence, axis=0))[0]

	            
	        #3. Viz logic
			if res[np.argmax(res)] > threshold: 
				if len(sentence) > 0: 
					if alphabets[np.argmax(res)] != sentence[-1]:
						sentence.append(alphabets[np.argmax(res)])
				else:
					sentence.append(alphabets[np.argmax(res)])

			if len(sentence) > 27: 
				sentence = sentence[-27:]
	            
		cv2.rectangle(image, (0,0), (640, 40), (224, 224, 224), -1)
		cv2.putText(image, ''.join(sentence), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
		__, frame = cv2.imencode('.jpg', image)
		yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')    


def translating_ytb():
	video = CamGear(source= "https://www.youtube.com/watch?v=VJj-DzfHSkQ", stream_mode = True, logging=True).start()
	threshold = 0.98
	words = []
	characters = []
	sequence = []
	frame_no = 0
	while True:
		frame = video.read()
		image, results = detecting_landmarks(frame)
		draw_connections(image, results)
		keypoints = get_coordinates(results)
		sequence.append(keypoints)
		sequence = sequence[-10:]
		if len(sequence) == 10:
			probs = model.predict(np.expand_dims(sequence, axis = 0))[0]
			pred = alphabets[np.argmax(probs)]
			if probs[ np.argmax(probs)] > threshold:
				if len(words) >0:
					if pred != words[-1]:
						words.append(pred)
				else:
					words.append(pred)
				if len(characters) >0:
					if words[-1] != characters[-1]:
						characters.append(words)
				else:
					characters.append(words)
			if len(words) >30:
				words = words[-30:]
			cv2.rectangle(image, (0,0), (1920, 40), (224, 224, 224), -1)
			cv2.putText(image, ''.join(words), (3,30), cv2.FONT_HERSHEY_SIMPLEX, 1 , (0,0,255), 2, cv2.LINE_AA)
			__, frame = cv2.imencode('.jpg', image)
			yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame.tobytes() + b'\r\n')







@app.route("/youtube")	
def youtube():
	return render_template("youtube.html")

@app.route("/ytb")
def ytb():
	return Response(translating_ytb(), mimetype='multipart/x-mixed-replace; boundary=frame')






@app.route('/webcam')
def webcam():

	return Response(translating_web(), mimetype='multipart/x-mixed-replace; boundary=frame')

	



if __name__ == "__main__":
	app.run(debug = True)
