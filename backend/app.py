from flask import Flask, Response, render_template, jsonify, request, send_from_directory
import cv2
import numpy as np
import os
import time
import pyttsx3
import threading
import enchant

# Initialize dictionary for word suggestions
dictionary = enchant.Dict("en-US")

# Import HandDetector from cvzone
try:
    from cvzone.HandTrackingModule import HandDetector
    # Initialize hand detectors
    hd = HandDetector(maxHands=1)
    hd2 = HandDetector(maxHands=1)
except ImportError:
    print("Error: cvzone not installed. Install with 'pip install cvzone'")
    hd = None
    hd2 = None

app = Flask(__name__, static_folder='../frontend', template_folder='../frontend')

# Global variables to store state
current_prediction = ""
current_symbol = "-"
current_sentence = ""
suggestions = ["", "", "", ""]
offset = 29

# Previous character tracking (similar to the GUI app)
prev_char = ""
count = -1
ten_prev_char = [" " for _ in range(10)]

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load model function
def load_model():
    print("Loading CNN model...")
    try:
        import tensorflow as tf
        model_path = os.path.join(os.path.dirname(__file__), 'model', 'cnn8grps_rad1_model.h5')
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Calculate distance between two points
def distance(x, y):
    return np.sqrt(((x[0] - y[0]) ** 2) + ((x[1] - y[1]) ** 2))

# Predict sign from frame using the same logic as the GUI app
def predict_sign(frame, model=None):
    global current_prediction, current_symbol, current_sentence, prev_char, count, ten_prev_char, suggestions
    
    if model is None or hd is None:
        # Return the current prediction if model isn't loaded
        return current_symbol, None
    
    try:
        # Find hands in the frame
        hands = hd.findHands(frame, draw=False, flipType=True)
        
        if hands[0]:
            hand = hands[0]
            handmap = hand[0]
            x, y, w, h = handmap['bbox']
            
            # Extract the hand region with offset
            hand_image = frame[max(0, y - offset):min(frame.shape[0], y + h + offset), 
                              max(0, x - offset):min(frame.shape[1], x + w + offset)]
            
            if hand_image.size == 0:
                return current_symbol, None
            
            # Create white background for hand landmarks
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255
            
            # Find hand landmarks in the extracted image
            handz = hd2.findHands(hand_image, draw=False, flipType=True)
            
            if handz[0]:
                hand = handz[0]
                handmap = hand[0]
                pts = handmap['lmList']
                
                # Draw hand landmarks on white background
                os_x = ((400 - w) // 2) - 15
                os_y = ((400 - h) // 2) - 15
                
                # Draw connections between landmarks (same as GUI app)
                for t in range(0, 4, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                             (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                for t in range(5, 8, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                             (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                for t in range(9, 12, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                             (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                for t in range(13, 16, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                             (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                for t in range(17, 20, 1):
                    cv2.line(white, (pts[t][0] + os_x, pts[t][1] + os_y), 
                             (pts[t + 1][0] + os_x, pts[t + 1][1] + os_y), (0, 255, 0), 3)
                
                # Connect palm landmarks
                cv2.line(white, (pts[5][0] + os_x, pts[5][1] + os_y), 
                         (pts[9][0] + os_x, pts[9][1] + os_y), (0, 255, 0), 3)
                cv2.line(white, (pts[9][0] + os_x, pts[9][1] + os_y), 
                         (pts[13][0] + os_x, pts[13][1] + os_y), (0, 255, 0), 3)
                cv2.line(white, (pts[13][0] + os_x, pts[13][1] + os_y), 
                         (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), 
                         (pts[5][0] + os_x, pts[5][1] + os_y), (0, 255, 0), 3)
                cv2.line(white, (pts[0][0] + os_x, pts[0][1] + os_y), 
                         (pts[17][0] + os_x, pts[17][1] + os_y), (0, 255, 0), 3)
                
                # Draw landmark points
                for i in range(21):
                    cv2.circle(white, (pts[i][0] + os_x, pts[i][1] + os_y), 2, (0, 0, 255), 1)
                
                # Prepare image for prediction
                white_resized = white.reshape(1, 400, 400, 3)
                
                # Make prediction
                prob = np.array(model.predict(white_resized, verbose=0)[0], dtype='float32')
                ch1 = np.argmax(prob, axis=0)
                prob[ch1] = 0
                ch2 = np.argmax(prob, axis=0)
                prob[ch2] = 0
                ch3 = np.argmax(prob, axis=0)
                prob[ch3] = 0
                
                pl = [ch1, ch2]
                
                # Implement the same prediction logic as in the GUI app
                # This is a simplified version of the complex logic in the original code
                
                # condition for [Aemnst]
                l = [[5, 2], [5, 3], [3, 5], [3, 6], [3, 0], [3, 2], [6, 4], [6, 1], [6, 2], [6, 6], [6, 7], [6, 0], [6, 5],
                     [4, 1], [1, 0], [1, 1], [6, 3], [1, 6], [5, 6], [5, 1], [4, 5], [1, 4], [1, 5], [2, 0], [2, 6], [4, 6],
                     [1, 0], [5, 7], [1, 6], [6, 1], [7, 6], [2, 5], [7, 1], [5, 4], [7, 0], [7, 5], [7, 2]]
                if pl in l:
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 0
                
                # condition for [o][s]
                l = [[2, 2], [2, 1]]
                if pl in l:
                    if (pts[5][0] < pts[4][0]):
                        ch1 = 0
                        print("++++++++++++++++++")
                
                # condition for [c0][aemnst]
                l = [[0, 0], [0, 6], [0, 2], [0, 5], [0, 1], [0, 7], [5, 2], [7, 6], [7, 1]]
                pl = [ch1, ch2]
                if pl in l:
                    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[4][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and 
                        pts[0][0] > pts[20][0]) and pts[5][0] > pts[4][0]:
                        ch1 = 2
                
                # condition for [c0][aemnst]
                l = [[6, 0], [6, 6], [6, 2]]
                pl = [ch1, ch2]
                if pl in l:
                    if distance(pts[8], pts[16]) < 52:
                        ch1 = 2
                
                # Map group index to character (simplified from original)
                if ch1 == 0:
                    ch1 = 'S'
                    if pts[4][0] < pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0]:
                        ch1 = 'A'
                    if pts[4][0] > pts[6][0] and pts[4][0] < pts[10][0] and pts[4][0] < pts[14][0] and pts[4][0] < pts[18][0] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]:
                        ch1 = 'T'
                    if pts[4][1] > pts[8][1] and pts[4][1] > pts[12][1] and pts[4][1] > pts[16][1] and pts[4][1] > pts[20][1]:
                        ch1 = 'E'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][0] > pts[14][0] and pts[4][1] < pts[18][1]:
                        ch1 = 'M'
                    if pts[4][0] > pts[6][0] and pts[4][0] > pts[10][0] and pts[4][1] < pts[18][1] and pts[4][1] < pts[14][1]:
                        ch1 = 'N'
                
                elif ch1 == 2:
                    if distance(pts[12], pts[4]) > 42:
                        ch1 = 'C'
                    else:
                        ch1 = 'O'
                
                elif ch1 == 3:
                    if (distance(pts[8], pts[12])) > 72:
                        ch1 = 'G'
                    else:
                        ch1 = 'H'
                
                elif ch1 == 7:
                    if distance(pts[8], pts[4]) > 42:
                        ch1 = 'Y'
                    else:
                        ch1 = 'J'
                
                elif ch1 == 4:
                    ch1 = 'L'
                
                elif ch1 == 6:
                    ch1 = 'X'
                
                elif ch1 == 5:
                    if pts[4][0] > pts[12][0] and pts[4][0] > pts[16][0] and pts[4][0] > pts[20][0]:
                        if pts[8][1] < pts[5][1]:
                            ch1 = 'Z'
                        else:
                            ch1 = 'Q'
                    else:
                        ch1 = 'P'
                
                elif ch1 == 1:
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'B'
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'D'
                    if (pts[6][1] < pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'F'
                    if (pts[6][1] < pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = 'I'
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] < pts[20][1]):
                        ch1 = 'W'
                    if (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] < pts[20][1]) and pts[4][1] < pts[9][1]:
                        ch1 = 'K'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) < 8) and (
                            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]):
                        ch1 = 'U'
                    if ((distance(pts[8], pts[12]) - distance(pts[6], pts[10])) >= 8) and (
                            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]) and (pts[4][1] > pts[9][1]):
                        ch1 = 'V'
                    if (pts[8][0] > pts[12][0]) and (
                            pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] <
                            pts[20][1]):
                        ch1 = 'R'
                
                # Check for space
                if ch1 == 1 or ch1 =='E' or ch1 =='S' or ch1 =='X' or ch1 =='Y' or ch1 =='B':
                    if (pts[6][1] > pts[8][1] and pts[10][1] < pts[12][1] and pts[14][1] < pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = " "
                
                # Check for next
                if ch1 == 'E' or ch1 == 'Y' or ch1 == 'B':
                    if (pts[4][0] < pts[5][0]) and (pts[6][1] > pts[8][1] and pts[10][1] > pts[12][1] and pts[14][1] > pts[16][1] and pts[18][1] > pts[20][1]):
                        ch1 = "next"
                
                # Check for backspace
                if ch1 == 'Next' or ch1 == 'B' or ch1 == 'C' or ch1 == 'H' or ch1 == 'F' or ch1 == 'X':
                    if (pts[0][0] > pts[8][0] and pts[0][0] > pts[12][0] and pts[0][0] > pts[16][0] and pts[0][0] > pts[20][0]) and (pts[4][1] < pts[8][1] and pts[4][1] < pts[12][1] and pts[4][1] < pts[16][1] and pts[4][1] < pts[20][1]) and (pts[4][1] < pts[6][1] and pts[4][1] < pts[10][1] and pts[4][1] < pts[14][1] and pts[4][1] < pts[18][1]):
                        ch1 = 'Backspace'
                
                # Handle special characters
                if ch1 == "next" and prev_char != "next":
                    if ten_prev_char[(count-2)%10] != "next":
                        if ten_prev_char[(count-2)%10] == "Backspace":
                            current_sentence = current_sentence[0:-1]
                        else:
                            if ten_prev_char[(count-2)%10] != "Backspace":
                                current_sentence = current_sentence + ten_prev_char[(count-2)%10]
                    else:
                        if ten_prev_char[(count-0)%10] != "Backspace":
                            current_sentence = current_sentence + ten_prev_char[(count-0)%10]
                
                if ch1 == "  " and prev_char != "  ":
                    current_sentence = current_sentence + "  "
                
                # Update character tracking
                prev_char = ch1
                current_symbol = ch1
                count += 1
                ten_prev_char[count%10] = ch1
                
                # Generate word suggestions
                if len(current_sentence.strip()) != 0:
                    st = current_sentence.rfind(" ")
                    ed = len(current_sentence)
                    word = current_sentence[st+1:ed]
                    
                    if len(word.strip()) != 0:
                        if dictionary.check(word):
                            suggs = dictionary.suggest(word)
                            lenn = len(suggs)
                            
                            suggestions = ["", "", "", ""]
                            if lenn >= 1:
                                suggestions[0] = suggs[0]
                            if lenn >= 2:
                                suggestions[1] = suggs[1]
                            if lenn >= 3:
                                suggestions[2] = suggs[2]
                            if lenn >= 4:
                                suggestions[3] = suggs[3]
                
                # Return the processed image for display
                return current_symbol, white
    
    except Exception as e:
        print(f"Prediction error: {e}")
    
    return current_symbol, None

# Video streaming generator function
def generate_frames():
    camera = cv2.VideoCapture(0)  # Use default camera
    
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return
    
    global current_symbol
    
    # Get the model
    model = load_model()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Flip the frame horizontally to match the GUI application
        frame = cv2.flip(frame, 1)
        
        # Make prediction with the model
        current_symbol, processed_frame = predict_sign(frame, model)
        
        # Draw prediction on frame
        cv2.putText(frame, f"Predicted: {current_symbol}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # If we have a processed frame (hand landmarks), show it in a corner
        if processed_frame is not None:
            # Resize the processed frame to be smaller
            small_processed = cv2.resize(processed_frame, (160, 160))
            # Place it in the top-right corner
            frame[10:170, frame.shape[1]-170:frame.shape[1]-10] = small_processed
        
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        # Yield the frame in byte format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('../frontend', path)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global current_symbol, current_sentence, suggestions
    
    return jsonify({
        'char': current_symbol,
        'sentence': current_sentence,
        'suggestions': suggestions
    })

@app.route('/speak', methods=['POST'])
def speak_text():
    global current_sentence
    
    if current_sentence:
        # Use a thread to avoid blocking the response
        def speak():
            # Create a new engine instance each time to avoid issues
            speak_engine = pyttsx3.init()
            speak_engine.say(current_sentence)
            speak_engine.runAndWait()
            # Explicitly stop and dispose of the engine
            speak_engine.stop()
        
        threading.Thread(target=speak).start()
    
    return jsonify({'status': 'speaking'})

@app.route('/clear', methods=['POST'])
def clear_text():
    global current_symbol, current_sentence, suggestions, prev_char, count, ten_prev_char
    
    current_symbol = "-"
    current_sentence = ""
    suggestions = ["", "", "", ""]
    prev_char = ""
    count = -1
    ten_prev_char = [" " for _ in range(10)]
    
    return jsonify({'status': 'cleared'})

@app.route('/update_sentence', methods=['POST'])
def update_sentence():
    global current_sentence
    
    data = request.json
    if 'sentence' in data:
        current_sentence = data['sentence']
    
    return jsonify({'status': 'updated', 'sentence': current_sentence})

if __name__ == '__main__':
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)
