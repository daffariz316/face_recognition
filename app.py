# import os
# import cv2
# import numpy as np
# import threading
# from flask import Flask, render_template, Response
# from sklearn.preprocessing import LabelEncoder
# from sklearn.neighbors import KNeighborsClassifier
# from deepface import DeepFace  

# app = Flask(__name__)

# dataset_path = r"C:\Users\VICTUS\Downloads\new project\py\python\dataset1"

# # Fungsi untuk memuat dataset wajah dan membuat embedding
# def load_dataset(dataset_path):
#     face_embeddings = []
#     face_labels = []

#     for category in os.listdir(dataset_path):
#         category_path = os.path.join(dataset_path, category)
#         if os.path.isdir(category_path):
#             for image_name in os.listdir(category_path):
#                 image_path = os.path.join(category_path, image_name)
#                 image = cv2.imread(image_path)
#                 if image is None:
#                     continue
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#                 face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#                 faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
#                 for (x, y, w, h) in faces:
#                     face = image[y:y+h, x:x+w]
#                     try:
#                         embedding = DeepFace.represent(face, model_name="SFace", detector_backend="opencv")[0]["embedding"]
#                         face_embeddings.append(embedding)
#                         face_labels.append(category)
#                     except:
#                         continue

#     return np.array(face_embeddings), np.array(face_labels)

# face_embeddings, face_labels = load_dataset(dataset_path)
# label_encoder = LabelEncoder()
# face_labels_encoded = label_encoder.fit_transform(face_labels)
# knn_classifier = KNeighborsClassifier(n_neighbors=3)
# knn_classifier.fit(face_embeddings, face_labels_encoded)

# video_capture = cv2.VideoCapture(0)

# def generate_frames():
#     while True:
#         ret, frame = video_capture.read()
#         if not ret:
#             break
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
#         threads = []
        
#         def recognize_face(face, x, y, w, h, frame):
#             try:
#                 embedding = DeepFace.represent(face, model_name="SFace", detector_backend="opencv")[0]["embedding"]
#                 embedding = np.array(embedding).reshape(1, -1)
#                 label_index = knn_classifier.predict(embedding)[0]
#                 label = label_encoder.inverse_transform([label_index])[0]
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                 cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
#             except:
#                 pass

#         for (x, y, w, h) in faces:
#             face = frame[y:y+h, x:x+w].copy()
#             thread = threading.Thread(target=recognize_face, args=(face, x, y, w, h, frame))
#             thread.start()
#             threads.append(thread)

#         for thread in threads:
#             thread.join()
        
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame_bytes = buffer.tobytes()
#         yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# @app.route('/')
# def index():
#     return render_template('index.html')
# # print("Templates:", os.listdir("templates"))

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(debug=True)
import os
import cv2
import numpy as np
import threading
from flask import Flask, render_template, Response
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from deepface import DeepFace  

app = Flask(__name__)

dataset_path = r"C:\Users\VICTUS\Downloads\new project\py\python\dataset1"

# Fungsi untuk memuat dataset wajah dan membuat embedding
def load_dataset(dataset_path):
    face_embeddings = []
    face_labels = []

    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            for image_name in os.listdir(category_path):
                image_path = os.path.join(category_path, image_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
                for (x, y, w, h) in faces:
                    face = image[y:y+h, x:x+w]
                    try:
                        embedding = DeepFace.represent(face, model_name="SFace", detector_backend="opencv")[0]["embedding"]
                        face_embeddings.append(embedding)
                        face_labels.append(category)
                    except:
                        continue

    return np.array(face_embeddings), np.array(face_labels)

face_embeddings, face_labels = load_dataset(dataset_path)
label_encoder = LabelEncoder()
face_labels_encoded = label_encoder.fit_transform(face_labels)
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(face_embeddings, face_labels_encoded)

video_capture = cv2.VideoCapture(0)
accuracy_list = []

def generate_frames():
    global accuracy_list

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30))
        threads = []
        
        def recognize_face(face, x, y, w, h, frame):
            global accuracy_list
            try:
                embedding = DeepFace.represent(face, model_name="SFace", detector_backend="opencv")[0]["embedding"]
                embedding = np.array(embedding).reshape(1, -1)

                # Prediksi label dan probabilitas
                label_index = knn_classifier.predict(embedding)[0]
                probabilities = knn_classifier.predict_proba(embedding)[0]
                confidence = max(probabilities) * 100  # Ambil probabilitas tertinggi
                
                label = label_encoder.inverse_transform([label_index])[0]
                
                # Simpan accuracy ke list
                accuracy_list.append(confidence)

                # Gambar kotak dan label di wajah
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except:
                pass

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w].copy()
            thread = threading.Thread(target=recognize_face, args=(face, x, y, w, h, frame))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
        
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    global accuracy_list
    avg_accuracy = round(sum(accuracy_list) / len(accuracy_list), 2) if accuracy_list else 0
    accuracy_list = []  # Reset agar tidak terus menumpuk
    return render_template('index.html', accuracy=avg_accuracy)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
