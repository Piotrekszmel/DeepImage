from crypt import methods
import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template


UPLOAD_FOLDER = './'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ['jpg', 'jpeg', 'png']

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = 'in.jpg'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            filename_out = 'out.jpg'
            # ffwd_to_img(filename, filename_out, request.form['checkpoint'])
            return redirect(url_for('uploaded_file',
                                    filename=filename))
    return render_template("index.html")

@app.route('/uploads<filename>', methods=["GET"])
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename, as_attachment=True)

if __name__ == "__main__":
    app.run(host="127.0.0.1")