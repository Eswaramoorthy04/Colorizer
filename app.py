from flask import Flask, request, render_template, redirect, send_from_directory
import os
import numpy as np
import cv2
import gdown

app = Flask(__name__, static_folder="static")

# Directories
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
MODEL_DIR = "models"

for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, MODEL_DIR]:
    os.makedirs(folder, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER

# Google Drive model files
FILES = {
    "prototxt": ("https://drive.google.com/uc?id=1NuxLxMh_vCMx-WE2a7UZIOribB2yEzfu", os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")),
    "caffemodel": ("https://drive.google.com/uc?id=1igk4lbedVo36COvSg1dLXsSthhNPq2yl", os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")),
    "points": ("https://drive.google.com/uc?id=11JvzsPf6imxMTKvUOU6zbDqVPJiEulOC", os.path.join(MODEL_DIR, "pts_in_hull.npy"))
}

# Download models if missing
for url, path in FILES.values():
    if not os.path.exists(path):
        print(f"Downloading {path}...")
        gdown.download(url, path, quiet=False)

# Load model
print("Loading colorization model...")
net = cv2.dnn.readNetFromCaffe(FILES["prototxt"][1], FILES["caffemodel"][1])
pts = np.load(FILES["points"][1])

# Set cluster centers
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId("class8_ab")).blobs = [pts.astype("float32")]
net.getLayer(net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, dtype="float32")]

# Route: Upload + Process
@app.route("/", methods=["GET", "POST"])
def index():
    colorized_image = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return redirect(request.url)

        # Save original
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)

        # Colorize
        image = cv2.imread(input_path)
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        L = cv2.resize(lab, (224, 224))[:, :, 0]
        L -= 50

        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = lab[:, :, 0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")

        # Save output
        output_filename = f"colorized_{file.filename}"
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        cv2.imwrite(output_path, colorized)

        colorized_image = output_filename

    return render_template("index.html", colorized_image=colorized_image)

# Download route
@app.route("/download/<filename>")
def download(filename):
    return send_from_directory(app.config["OUTPUT_FOLDER"], filename, as_attachment=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
