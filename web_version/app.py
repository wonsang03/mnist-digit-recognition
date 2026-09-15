"""Web handwritten digit recognizer (Flask).

Created: 2025-09-15
Serves a browser canvas and exposes a JSON prediction endpoint that reuses the
exact same CNN and preprocessing pipeline as the desktop version.

Run:
    python web_version/app.py
Then open the address printed in the terminal (default: http://localhost:5000).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

# Allow running this file directly from the web_version folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.model import MODEL_PATH, load_model  # noqa: E402
from common.preprocess import has_ink, predict_digit  # noqa: E402

app = Flask(__name__)

# Reject oversized uploads: a 280x280 PNG data URL is only a few kilobytes.
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

# The model is loaded once at start-up and reused for every request.
model = None


def get_model():
    """Return the cached model, loading it on the first call."""
    global model
    if model is None:
        model = load_model()
    return model


def decode_data_url(data_url: str) -> Image.Image:
    """Turn a 'data:image/png;base64,...' string into a PIL image."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    return Image.open(io.BytesIO(raw))


@app.route("/")
def index():
    """Render the drawing page."""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Recognize the digit sent as a base64 PNG data URL."""
    payload = request.get_json(silent=True) or {}
    data_url = payload.get("image")

    if not data_url:
        return jsonify({"error": "No image was sent."}), 400

    try:
        image = decode_data_url(data_url)
    except (binascii.Error, ValueError, UnidentifiedImageError):
        return jsonify({"error": "The image could not be decoded."}), 400

    # The browser canvas is transparent where nothing was drawn, so it is
    # flattened onto a white background first to match the desktop input.
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image)

    # A blank canvas still yields a confident-looking prediction, so it is
    # rejected here rather than reported as a digit.
    if not has_ink(image):
        return jsonify({"error": "The canvas is empty. Draw a digit first."}), 400

    try:
        digit, confidence, scores, processed = predict_digit(get_model(), image)
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503

    # Send the 28x28 network input back so the page can show what the model saw.
    buffer = io.BytesIO()
    processed.save(buffer, format="PNG")
    preview = base64.b64encode(buffer.getvalue()).decode("ascii")

    return jsonify(
        {
            "digit": digit,
            "confidence": round(confidence, 2),
            "scores": [round(value, 2) for value in scores],
            "preview": f"data:image/png;base64,{preview}",
        }
    )


@app.route("/health")
def health():
    """Simple status endpoint that reports whether the model is available."""
    return jsonify({"status": "ok", "model_ready": MODEL_PATH.exists()})


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the web digit recognizer.")
    parser.add_argument("--host", default="127.0.0.1", help="host to bind to")
    parser.add_argument("--port", type=int, default=5000, help="port to bind to")
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        print(f"Trained model not found at '{MODEL_PATH}'.")
        print("Run 'python train_model.py' in the project root first.")
        sys.exit(1)

    # Fail fast at start-up instead of on the first request.
    get_model()

    print("=" * 58)
    print("  Handwritten Digit Recognition - Web Version")
    print(f"  Open http://localhost:{args.port} in your browser")
    print("  Press Ctrl+C to stop the server")
    print("=" * 58)

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
