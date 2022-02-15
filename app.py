from flask import Flask, request, redirect, url_for, send_from_directory, render_template
import os

from deepimage.core.style_transfer.photo_demo import predict, make_photo

UPLOAD_FOLDER = "./deepimage"


def allowed_file(filename: str):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ["jpg", "jpeg"]

app = Flask(__name__, template_folder="deepimage/templates", static_folder="deepimage/static")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

Config = {
    "model": "deepimage/core/style_transfer/models/21styles.model",
    "style_folder": "deepimage/static/images/styles",
    "style_size": 512,
    "demo_size": 480,
    "output_image": "output.jpg",
    "mirror": False,
    "if_download": 0,
    "style_idx": 0,
    "ngf": 128,
    "content_image": "content.jpg",
    "save_path": os.path.join(app.config["UPLOAD_FOLDER"], "content.jpg")
}


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if request.form.get("checkpoint"):
            Config["style_idx"] = int(request.form["checkpoint"])
        if request.form.get("mirror"):
            Config["mirror"] = True
        if request.form.get("if_download"):
            Config["if_download"] = 1
        if request.form.get("resize"):
            Config["resize"] = True
            if request.form.get("new_height"):
                Config["new_height"] = int(request.form.get("new_height"))
            if request.form.get("new_width"):
                Config["new_width"] = int(request.form.get("new_width"))   
        if "file" in request.files:
            file = request.files["file"]
            if file.filename != "" and allowed_file(file.filename):
                file.save(Config["save_path"])
                predict(Config)
                return redirect(url_for("output", filename=Config["output_image"], if_download=Config["if_download"]))
            else:
                result = make_photo(Config["save_path"])
                if result:
                    predict(Config)
                    return redirect(url_for("output", filename=Config["output_image"], if_download=Config["if_download"]))
        return redirect(request.url)
    
    return render_template("index.html")


@app.route("/output/<filename>/<if_download>", methods=["GET"])
def output(filename: str, if_download: str):
    return send_from_directory(app.config["UPLOAD_FOLDER"],
                               filename, as_attachment=int(if_download))


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0')