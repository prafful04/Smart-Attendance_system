from flask import Flask, render_template, redirect, send_file, url_for, request
import os
from camera import start_camera
from attendance import recognize_faces, recognize_faces_from_image  # ✅ ADD THIS
from train_model import train_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/start', methods=['POST'])
def start():
    name = request.form.get('username')
    if not name:
        return "No name provided", 400
    start_camera()
    return redirect(url_for('home'))


@app.route('/mark', methods=['POST'])
def mark():
    print("Marking attendance...")
    recognize_faces()
    return redirect(url_for('home'))

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No image uploaded", 400

    file = request.files['image']
    if file.filename == '':
        return "No file selected", 400

    os.makedirs('uploads', exist_ok=True)
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)

    recognize_faces_from_image(filepath)
    return redirect(url_for('home'))


@app.route('/view')
def view_attendance():
    import pandas as pd
    try:
        df = pd.read_csv('attendance.csv')
        return df.to_html(classes='table table-striped', index=False)
    except FileNotFoundError:
        return "<h3>No attendance data found.</h3>"
@app.route('/train', methods=['POST'])
def train():
    train_model()
    return redirect(url_for('home'))

@app.route('/download')
def download():
    return send_file('attendance.csv', as_attachment=True)


@app.route('/exit')
def exit_app():
    return "<h2>Goodbye! (You can close the browser)</h2>"

if __name__ == '__main__':
  app.run(debug=False, host='0.0.0.0', port=5000)

