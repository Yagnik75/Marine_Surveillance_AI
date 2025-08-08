import sys
import os
import subprocess
from flask import Flask, send_from_directory

# Serve files from the current directory
app = Flask(__name__, static_folder=".", static_url_path="")


@app.route("/")
def serve_index():
    return send_from_directory(".", "index2.html")


@app.route("/run-ship", methods=["POST"])
def run_ship():
    subprocess.Popen(
        [sys.executable, "ship3.py"],
        cwd=os.path.join(os.getcwd(), "ship_detection_folder"),
    )
    return "", 204


@app.route("/run-oil", methods=["POST"])
def run_oil():
    subprocess.Popen(
        [sys.executable, "oil2final.py"],
        cwd=os.path.join(os.getcwd(), "oil_spill_detector_folder"),
    )
    return "", 204


@app.route("/run-debris", methods=["POST"])
def run_debris():
    subprocess.Popen(
        [sys.executable, "deb8.py"], cwd=os.path.join(os.getcwd(), "marinedebris")
    )
    return "", 204


if __name__ == "__main__":
    app.run(debug=True)
