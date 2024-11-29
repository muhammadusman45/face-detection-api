from flask import Flask, request, jsonify
from flask_restx import Api, Resource
import face_recognition
import os
from PIL import Image
import io
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
api = Api(app, doc='/docs')
ns = api.namespace('hello', description='Hello World operations')

ATTENDANCE_DIR = 'attendance_records'
os.makedirs(ATTENDANCE_DIR, exist_ok=True)

IMAGE_DIR = 'uploaded_images'
os.makedirs(IMAGE_DIR, exist_ok=True)

known_face_encodings = []
known_face_names = []

# Function to load known faces
def load_known_faces():
    known_face_encodings.clear()  # Clear previous encodings
    known_face_names.clear()      # Clear previous names

    for filename in os.listdir(IMAGE_DIR):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
            file_path = os.path.join(IMAGE_DIR, filename)
            # print( "file_path --->", file_path)
            image = face_recognition.load_image_file(file_path)
            face_encodings = face_recognition.face_encodings(image)
            # print("images ----->" ,face_encodings)
            if face_encodings:  
                face_encoding = face_encodings[0]  
                name = os.path.splitext(filename)[0]  # Get the file name without extension
                known_face_encodings.append(face_encoding)
                known_face_names.append(name)
            else:
                print(f"No face found in {file_path}")
# load_known_faces()
# Function to save attendance
def save_attendance(name):
    with open(os.path.join(ATTENDANCE_DIR, f"{name}.txt"), 'a') as file:
        file.write(f"{name} attended\n")

@ns.route('/')
class HelloWorld(Resource):
    def get(self):
        """Returns a Hello, World! message"""
        return {'message': 'Hello, World!'}

    def post(self):
        """Receives text and echoes it back"""
        data = request.get_json()
        if 'text' in data:
            return {'received_text': data['text']}
        else:
            return {'error': 'No text provided'}, 400

# Route to handle image uploads for face recognition
@app.route('/upload', methods=['POST'])
def image_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image = Image.open(io.BytesIO(file.read()))
    image = image.convert('RGB')  
    image_np = np.array(image)

    face_locations = face_recognition.face_locations(image_np)
    face_encodings = face_recognition.face_encodings(image_np, face_locations)
    
    tolerance = 0.4
    message = "No faces found in the image"  # Default message

    if face_encodings:
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=tolerance)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                save_attendance(name)
                message = f"{name}"
                break  # Exit loop after first match
            else:
                message = "No known face matched, attendance not recorded"
    else:
        message = "No faces found in the image"

    return jsonify({'status': message}), 200

# Route to handle image uploads and update known faces
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded image to the 'uploaded_images' directory
    image_path = os.path.join(IMAGE_DIR, file.filename)
    file.save(image_path)

    # Reload the known faces after the new image is uploaded
    load_known_faces()

    return jsonify({'message': 'Image uploaded successfully!', 'image_path': image_path}), 200

if __name__ == '__main__':
    load_known_faces()  # Load known faces when the app starts
    app.run(host='0.0.0.0', port=5555, debug=True)
