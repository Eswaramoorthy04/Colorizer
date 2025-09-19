from flask import Flask, request, render_template, redirect, send_from_directory
import os
import numpy as np
import cv2
import gdown

app = Flask(__name__, static_folder='static')

# Folders
UPLOAD_FOLDER = 'static/uploads'
MODEL_FOLDER = 'model'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(MODEL_FOLDER):
    os.makedirs(MODEL_FOLDER)

# Google Drive file IDs
GDRIVE_FILES = {
    "prototxt": "1NuxLxMh_vCMx-WE2a7UZIOribB2yEzfu",
    "caffemodel": "1igk4lbedVo36COvSg1dLXsSthhNPq2yl",
    "pts_in_hull": "11JvzsPf6imxMTKvUOU6zbDqVPJiEulOC"
}

# Download model files if not present
for name, file_id in GDRIVE_FILES.items():
    ext = ".prototxt" if name=="prototxt" else ".caffemodel" if name=="caffemodel" else ".npy"
    file_path = os.path.join(MODEL_FOLDER, name+ext)
    if not os.path.exists(file_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading {name} from Google Drive...")
        gdown.download(url, file_path, quiet=False)

# Paths
PROTOTXT = os.path.join(MODEL_FOLDER, "prototxt.prototxt")
MODEL = os.path.join(MODEL_FOLDER, "caffemodel.caffemodel")
POINTS = os.path.join(MODEL_FOLDER, "pts_in_hull.npy")

# Load model
print("Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Maximum output size
MAX_WIDTH = 1024
MAX_HEIGHT = 1024

@app.route("/", methods=["GET", "POST"])
def upload_file():
    colorized_image = None
    if request.method == "POST":
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        # Save uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Load original image
        image = cv2.imread(file_path)
        h_orig, w_orig = image.shape[:2]

        # Upscale very small images
        if min(h_orig, w_orig) < 224:
            scale_up = 224 / min(h_orig, w_orig)
            image = cv2.resize(image, (int(w_orig*scale_up), int(h_orig*scale_up)), interpolation=cv2.INTER_CUBIC)
            h_orig, w_orig = image.shape[:2]

        # Constrain very large images
        scale = min(MAX_WIDTH / w_orig, MAX_HEIGHT / h_orig, 1.0)
        new_w, new_h = int(w_orig * scale), int(h_orig * scale)
        image_resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Normalize and convert to LAB
        scaled = image_resized.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

        # Resize for model input (224x224)
        lab_rs = cv2.resize(lab, (224, 224))
        L = cv2.split(lab_rs)[0]
        L -= 50

        # Run model
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

        # Resize ab to output image size
        ab_resized = cv2.resize(ab, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        # Combine L channel with ab channels
        L_orig = cv2.split(lab)[0]
        colorized = np.concatenate((L_orig[:, :, np.newaxis], ab_resized), axis=2)

        # Convert to BGR and clip
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        # Save colorized image
        colorized_file_path = os.path.join(app.config['UPLOAD_FOLDER'], "colorized_" + file.filename)
        cv2.imwrite(colorized_file_path, colorized)
        colorized_image = "colorized_" + file.filename

    return render_template("index.html", colorized_image=colorized_image)

# Download route
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)
