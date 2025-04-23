# Sign2Text: Sign Language to Text Conversion

## Overview

Sign2Text is a web-based application that converts American Sign Language (ASL) gestures into text in real-time. The application uses computer vision and machine learning to detect hand gestures and translate them into corresponding letters and words, making communication more accessible for sign language users.

![Sign2Text Demo](demo/demo-placeholder.gif)


https://github.com/user-attachments/assets/586e05be-bebe-42f5-bd89-eb34b08cc399


## Project Background

More than 70 million deaf people around the world use sign languages to communicate. Sign language allows them to learn, work, access services, and be included in their communities. However, it's challenging to make everyone learn sign language to ensure people with disabilities can enjoy their rights on an equal basis with others.

This project aims to develop a user-friendly human-computer interface (HCI) where the computer understands American Sign Language, helping deaf and mute people communicate more easily with the wider community.

## Features

- **Real-time Sign Language Detection**: Converts ASL hand gestures to text in real-time
- **Word Suggestions**: Provides word suggestions based on detected characters
- **Text-to-Speech**: Speaks the detected sentence aloud
- **ASL Reference Chart**: Includes a reference chart for ASL alphabet
- **User-friendly Interface**: Clean, responsive web interface
- **Special Gestures**: Support for "space" and "next" gestures to build sentences
- **Background-Independent**: Works in various lighting conditions and backgrounds

## How It Works

### Innovative Approach to Hand Detection

The application uses a unique approach to detect hand gestures:

1. **Hand Landmark Detection**: Using MediaPipe to detect 21 key points on the hand
2. **White Background Rendering**: Drawing these landmarks on a plain white background
   - This eliminates issues with varying lighting conditions and backgrounds
3. **CNN Classification**: The hand gesture is classified using a Convolutional Neural Network
4. **Post-processing**: Additional logic determines the exact letter

### Special Gestures

The application recognizes two special gestures:

1. **Space Gesture**: Used to add a space between words
   - Make this gesture when you want to start a new word

2. **Next Gesture**: Used after making a letter gesture to add it to the sentence
   - This is essential for building words letter by letter

3. **Backspace Gesture**: Used to delete the last character
   - Useful for correcting mistakes

### Model Architecture

The CNN model was trained on 180 skeleton images of ASL alphabets. To improve accuracy, the 26 alphabets were divided into 8 classes of similar gestures:

1. [A, E, M, N, S, T]
2. [B, D, F, I, U, V, K, R, W]
3. [C, O]
4. [G, H]
5. [L]
6. [P, Q, Z]
7. [X]
8. [Y, J]

This approach achieved a 88-95% accuracy rate, even with varying backgrounds and lighting conditions.

## Usage

1. Position your hand in the camera view
2. Make ASL gestures for letters
3. Use the "next" gesture to add the detected character to the sentence
4. Alternatively, use the "Add to Sentence" button
5. Click on word suggestions to complete words
6. Use the "Speak" button to hear the sentence
7. Use the "Clear" button to start over

## Technical Details

### Backend

- **Flask**: Web framework for the application
- **OpenCV**: For webcam access and image processing
- **TensorFlow/Keras**: For loading and running the CNN model
- **MediaPipe/cvzone**: For hand landmark detection
- **pyttsx3**: For text-to-speech functionality
- **Enchant**: For word suggestions

### Frontend

- **HTML/CSS/JavaScript**: For the web interface
- **Fetch API**: For communication with the backend

## Advantages of Our Approach

1. **Background Independence**: Works in any environment, not just clean backgrounds
2. **Lighting Robustness**: Functions in various lighting conditions
3. **High Accuracy**: 88-95% accuracy in recognizing ASL alphabets
4. **Real-time Performance**: Fast enough for practical use
5. **User-friendly Interface**: Easy to use with minimal training

## Future Work

- Develop a mobile application version
- Expand to include more sign language gestures beyond the alphabet
- Implement continuous sign language recognition for common phrases
- Add support for other sign languages beyond ASL

## Credits and Acknowledgments

This project is based on the work of Devansh Raval (19IT470) and Kelvin Parmar (19IT473) under the guidance of Dr. Nilesh B. Prajapati at Birla Vishvakarma Mahavidyalaya Engineering College, Information Technology Department. 
(https://github.com/Devansh-47/Sign-Language-To-Text-and-Speech-Conversion)

The original research implemented:
- A CNN-based approach for sign language recognition
- MediaPipe for hand landmark detection
- An innovative technique of drawing landmarks on a white background to overcome lighting and background issues

Their approach achieved 97% accuracy in recognizing ASL alphabets, even in varying conditions.


---

Created by Tejashya Mehta
