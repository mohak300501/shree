
"""
  _________.__                          
 /   _____/|  |_________   ____   ____  
 \_____  \ |  |  \_  __ \_/ __ \_/ __ \ 
 /        \|   Y  \  | \/\  ___/\  ___/ 
/_______  /|___|  /__|    \___  >\___  >
        \/      \/            \/     \/ 

"""

from flask import Flask, render_template, request, redirect, url_for, flash, session, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
from datetime import datetime
from pathlib import Path
from ocrer import process_pdf, convert_to_iast, get_progress
import os, dotenv, threading

dotenv.load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "dev_fallback_key_123456")

# Make sure the pdfs directory exists
PDF_FOLDER = os.path.join(os.getcwd(), "pdfs")
os.makedirs(PDF_FOLDER, exist_ok=True)
app.config["PDF_FOLDER"] = PDF_FOLDER
ALLOWED_EXTENSIONS = {"pdf"}

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# ─────────────────────────── APP ROUTES ───────────────────────────

@app.context_processor
def inject_current_year():
    return {"current_year": datetime.now().year}

@app.route("/")
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return render_template("index.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("index"))
    if request.method == "POST":
        password = request.form["password"]
        if password == os.environ.get("APP_PASSWORD", "admin123"):
            session["logged_in"] = True
            flash("Welcome! You are logged in.", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid password.", "error")
            return redirect(url_for("login"))
    return render_template("login.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part.", "error")
            return redirect(url_for("upload"))

        file = request.files["file"]

        if file.filename == "":
            flash("No selected file.", "error")
            return redirect(url_for("upload"))

        if file and allowed_file(file.filename):
            filename = transliterate(file.filename, DEVANAGARI, ITRANS)
            filename = secure_filename(filename)
            file.save(os.path.join(app.config["PDF_FOLDER"], filename))
            flash(f"File '{filename}' uploaded successfully.", "success")
            return redirect(url_for("upload"))
        else:
            flash("Only PDF files are allowed.", "error")
            return redirect(url_for("upload"))

    # List all PDF files
    pdf_files = [f for f in os.listdir(app.config["PDF_FOLDER"]) if f.lower().endswith(".pdf")]

    return render_template("upload.html", pdf_files=pdf_files)

@app.route("/pdfs/<path:pdfname>")
def serve_pdf(pdfname):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    return send_from_directory(app.config["PDF_FOLDER"], pdfname)

@app.route("/deva/<path:pdfname>")
def serve_deva(pdfname):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    base_name = os.path.splitext(pdfname)[0]
    deva_folder = os.path.join("OCR_deva", base_name)

    if not os.path.exists(deva_folder):
        return "<h3>No OCR results found for this PDF.</h3>"

    # Sort .txt files numerically if named like page1.txt, page2.txt, etc.
    txt_files = sorted(
        [f for f in os.listdir(deva_folder) if f.lower().endswith(".txt")]
    )

    content = ""
    for txt_file in txt_files:
        txt_path = os.path.join(deva_folder, txt_file)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                page_text = f.read()
            content += f"<h4>{txt_file}</h4><pre>{page_text}</pre><hr>"
        except Exception as e:
            content += f"<p>Error reading {txt_file}: {e}</p>"

    return render_template("ocr_res.html", content=content)
 
@app.route("/iast/<path:pdfname>")
def serve_iast(pdfname):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    base_name = os.path.splitext(pdfname)[0]
    iast_folder = os.path.join("OCR_iast", base_name)

    if not os.path.exists(iast_folder):
        return "<h3>No OCR results found for this PDF.</h3>"

    # Sort .txt files numerically if named like page1.txt, page2.txt, etc.
    txt_files = sorted(
        [f for f in os.listdir(iast_folder) if f.lower().endswith(".txt")]
    )

    content = ""
    for txt_file in txt_files:
        txt_path = os.path.join(iast_folder, txt_file)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                page_text = f.read()
            content += f"<h4>{txt_file}</h4><pre>{page_text}</pre><hr>"
        except Exception as e:
            content += f"<p>Error reading {txt_file}: {e}</p>"

    return render_template("ocr_res.html", content=content)

@app.route("/pocr/<path:pdfname>")
def serve_pocr(pdfname):
    if not session.get("logged_in"):
        return redirect(url_for("login"))
    base_name = os.path.splitext(pdfname)[0]
    pocr_folder = os.path.join("PostOCR", base_name)

    if not os.path.exists(pocr_folder):
        return "<h3>No OCR results found for this PDF.</h3>"

    # Sort .txt files numerically if named like page1.txt, page2.txt, etc.
    txt_files = sorted(
        [f for f in os.listdir(pocr_folder) if f.lower().endswith(".txt")]
    )

    content = ""
    for txt_file in txt_files:
        txt_path = os.path.join(pocr_folder, txt_file)
        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                page_text = f.read()
            content += f"<h4>{txt_file}</h4><pre>{page_text}</pre><hr>"
        except Exception as e:
            content += f"<p>Error reading {txt_file}: {e}</p>"

    return render_template("ocr_res.html", content=content)

@app.route("/ocr/<path:pdfname>", methods=["POST"])
def ocr(pdfname):
    if not session.get("logged_in"):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        pdf_path = Path(app.config["PDF_FOLDER"]) / pdfname

        thread = threading.Thread(target=lambda: process_pdf(pdf_path))
        thread.daemon = True
        thread.start()

        return jsonify({"status": "started"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/ocr_progress/<path:pdfname>")
def ocr_progress(pdfname):
    if not session.get("logged_in"):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        return jsonify(get_progress())
            
    except Exception as e:
        return jsonify({"status": "error", "message": f"{pdfname}:{str(e)}"}), 500

@app.route("/iast/<path:pdfname>", methods=["POST"])
def iast(pdfname):
    if not session.get("logged_in"):
        return jsonify({"status": "error", "message": "Not authenticated"}), 401
    
    try:
        pdf_path = Path(app.config["PDF_FOLDER"]) / pdfname

        thread = threading.Thread(target=lambda: convert_to_iast(pdf_path))
        thread.daemon = True
        thread.start()

        return jsonify({"status": "started"})

    except Exception as e:
        return jsonify({"status": "error", "message": f"{pdfname}:{str(e)}"}), 500

@app.route("/viewpdf", methods=["GET", "POST"])
def viewpdf():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    # List all PDF files
    pdf_files = [f for f in os.listdir(app.config["PDF_FOLDER"]) if f.lower().endswith(".pdf")]

    return render_template("viewpdf.html", pdf_files=pdf_files)

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/logout", methods=["POST"])
def logout():
    session.pop("logged_in", None)
    flash("You have been logged out.", "success")
    return redirect(url_for("login"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

