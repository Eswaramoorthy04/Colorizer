from flask import Flask, request, render_template, redirect, send_from_directory, url_for
import os
import threading
import numpy as np
import cv2
import gdown

app = Flask(__name__, static_folder="static", template_folder="templates")

# ------------------------
# Simple configuration
# ------------------------
UPLOAD_DIR = os.path.join(app.static_folder, "uploads")
OUTPUT_DIR = os.path.join(app.static_folder, "outputs")
MODEL_DIR = "model"                 # local folder to store downloaded models
MAX_DIM = 1024                      # limit very large images
MIN_DIM_FOR_MODEL = 224             # upscale images smaller than this

for d in (UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR):
    os.makedirs(d, exist_ok=True)

# Google Drive download links (direct "uc?id=" links you gave)
PROTOTXT_URL = "https://drive.google.com/uc?id=1NuxLxMh_vCMx-WE2a7UZIOribB2yEzfu"
MODEL_URL = "https://drive.google.com/uc?id=1igk4lbedVo36COvSg1dLXsSthhNPq2yl"
POINTS_URL = "https://drive.google.com/uc?id=11JvzsPf6imxMTKvUOU6zbDqVPJiEulOC"

PROTOTXT_PATH = os.path.join(MODEL_DIR, "colorization_deploy_v2.prototxt")
MODEL_PATH = os.path.join(MODEL_DIR, "colorization_release_v2.caffemodel")
POINTS_PATH = os.path.join(MODEL_DIR, "pts_in_hull.npy")

# Globals for model state
_model_lock = threading.Lock()
_net = None
_model_loaded = False

def ensure_model_loaded():
    """
    Downloads model files from Google Drive if missing and loads the OpenCV DNN.
    This is called lazily (first time an image is processed) and is thread-safe.
    """
    global _net, _model_loaded

    if _model_loaded and _net is not None:
        return True

    with _model_lock:
        if _model_loaded and _net is not None:
            return True

        try:
            # Download files if missing
            if not os.path.exists(PROTOTXT_PATH):
                print("[MODEL] Downloading prototxt...")
                gdown.download(PROTOTXT_URL, PROTOTXT_PATH, quiet=False)
            if not os.path.exists(MODEL_PATH):
                print("[MODEL] Downloading caffemodel (this may take a while)...")
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            if not os.path.exists(POINTS_PATH):
                print("[MODEL] Downloading pts_in_hull.npy...")
                gdown.download(POINTS_URL, POINTS_PATH, quiet=False)

            print("[MODEL] Loading OpenCV DNN from:", PROTOTXT_PATH, MODEL_PATH)
            net = cv2.dnn.readNetFromCaffe(PROTOTXT_PATH, MODEL_PATH)

            # load points and set layer blobs (same as your local code)
            pts = np.load(POINTS_PATH)
            pts = pts.transpose().reshape(2, 313, 1, 1)
            class8 = net.getLayerId("class8_ab")
            conv8 = net.getLayerId("conv8_313_rh")
            net.getLayer(class8).blobs = [pts.astype("float32")]
            net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

            _net = net
            _model_loaded = True
            print("[MODEL] Loaded successfully.")
            return True
        except Exception as e:
            print("[MODEL] Failed to download/load model:", e)
            _net = None
            _model_loaded = False
            return False

def colorize_with_net(image, net):
    """
    image: BGR numpy image (H,W,3) uint8
    net: loaded OpenCV DNN
    returns colorized BGR uint8 image (H,W,3)
    """
    # normalize and convert to LAB
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # handle small images: upscale to avoid huge artifacts
    h_orig, w_orig = image.shape[:2]
    if min(h_orig, w_orig) < MIN_DIM_FOR_MODEL:
        scale_up = MIN_DIM_FOR_MODEL / min(h_orig, w_orig)
        image = cv2.resize(image, (int(w_orig * scale_up), int(h_orig * scale_up)), interpolation=cv2.INTER_CUBIC)
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        h_orig, w_orig = image.shape[:2]

    # constrain very large images
    scale = min(MAX_DIM / w_orig, MAX_DIM / h_orig, 1.0)
    new_w, new_h = int(w_orig * scale), int(h_orig * scale)
    if (new_w, new_h) != (w_orig, h_orig):
        image_rs = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        scaled = image_rs.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
    else:
        image_rs = image

    # prepare input for model
    lab_rs_for_net = cv2.resize(lab, (224, 224))
    L = cv2.split(lab_rs_for_net)[0]
    L -= 50

    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # resize ab to output size (image_rs)
    ab_rs = cv2.resize(ab, (image_rs.shape[1], image_rs.shape[0]), interpolation=cv2.INTER_CUBIC)

    L_orig = cv2.split(lab)[0]
    colorized = np.concatenate((L_orig[:, :, np.newaxis], ab_rs), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)
    colorized = (255 * colorized).astype("uint8")

    # if we resized up or down, colorized matches image_rs; if we changed size from original small->upscale,
    # we keep the colorized at image_rs size (safer visually). That's what we've chosen to do.
    return colorized

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def upload_file():
    colorized_filename = None
    error = None

    if request.method == "POST":
        # ensure model is ready (lazy load)
        ok = ensure_model_loaded()
        if not ok:
            error = "Model not available (download or load failed). Check logs."
            return render_template("index.html", colorized_image=None, error=error)

        # get file
        if 'file' not in request.files:
            return redirect(request.url)
        f = request.files['file']
        if f.filename == "":
            return redirect(request.url)

        # save input
        input_path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(input_path)

        try:
            image = cv2.imread(input_path)
            if image is None:
                raise ValueError("cv2.imread returned None (unsupported file or corrupt).")

            # run colorization
            colorized = colorize_with_net(image, _net)

            # save output
            colorized_filename = "colorized_" + f.filename
            output_path = os.path.join(OUTPUT_DIR, colorized_filename)
            cv2.imwrite(output_path, colorized)

            # show the colorized image from static/outputs
            return render_template("index.html", colorized_image=colorized_filename, error=None)
        except Exception as e:
            print("[PROCESS] Error during colorization:", e)
            error = "Error processing image. Check server logs."
            return render_template("index.html", colorized_image=None, error=error)

    return render_template("index.html", colorized_image=None, error=None)

@app.route("/download/<filename>")
def download_file(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)

@app.route("/health")
def health():
    return "OK", 200

# ------------------------
# Run
# ------------------------
if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    # debug=False for production
    app.run(host="0.0.0.0", port=port, debug=False)
