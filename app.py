from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import os
import numpy as np
import cv2
import gdown

app = Flask(__name__, static_folder='static')

# Upload folder
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Model folder
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive links for models
PROTOTXT_URL = "https://drive.google.com/uc?id=1NuxLxMh_vCMx-WE2a7UZIOribB2yEzfu"
MODEL_URL = "https://drive.google.com/uc?id=1igk4lbedVo36COvSg1dLXsSthhNPq2yl"
POINTS_URL = "https://drive.google.com/uc?id=11JvzsPf6imxMTKvUOU6zbDqVPJiEulOC"

PROTOTXT_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Download model files if not present
if not os.path.exists(PROTOTXT_PATH):
    gdown.download(PROTOTXT_URL, PROTOTXT_PATH, quiet=False)
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
if not os.path.exists(POINTS_PATH):
    gdown.download(POINTS_URL, POINTS_PATH, quiet=False)

# Load the colorization model
print("Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)
pts = np.load(POINTS_PATH)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

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

        # Read and colorize image
        image = cv2.imread(file_path)
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        # Save colorized image
        colorized_filename = "colorized_" + file.filename
        colorized_file_path = os.path.join(app.config['UPLOAD_FOLDER'], colorized_filename)
        cv2.imwrite(colorized_file_path, colorized)
        colorized_image = colorized_filename

    return render_template("index.html", colorized_image=colorized_image)

# Route to download the colorized image
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
