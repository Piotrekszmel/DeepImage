import os
import re
from flask import Flask, request, redirect, url_for, send_from_directory, render_template

from core.style_transfer.photo_demo import predict

UPLOAD_FOLDER = "./"

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ["jpg", "jpeg", "png"]

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    args = {
        "ngf": 128,
        "style_idx": 0,
        "model": "core/style_transfer/models/21styles.model",
        "style_folder": "static/images/styles",
        "style_size": 512,
        "demo_size": 480,
        "output_image": "output.jpg",
        "content_image": "content.jpg",
        "mirror": False,
    }
     
    if request.method == "POST":
        if request.form.get("mirror"):
            args["mirror"] = True
        if request.form.get("resize"):
            print("\n\n")
            print(type(request.form))
            args["resize"] = True
            if request.form.get("new_height"):
                args["new_height"] = int(request.form.get("new_height"))
            if request.form.get("new_width"):
                args["new_width"] = int(request.form.get("new_width"))   
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "" and allowed_file(file.filename):
                args["style_idx"] = request.form["checkpoint"]
                file.save(os.path.join(app.config["UPLOAD_FOLDER"], args["content_image"]))
                predict(args)
                return redirect(url_for("output", filename=args["output_image"]))
        return redirect(request.url)
    
    return render_template("index.html")

@app.route("/output/<filename>", methods=["GET"])
def output(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"],
                               filename)

if __name__ == "__main__":
    app.run(host="127.0.0.1")