import subprocess, time, os, shutil, multiprocessing, queue, io, dotenv, threading
from indic_transliteration.sanscript import transliterate, DEVANAGARI, IAST
from pathlib import Path
from google.cloud import vision

dotenv.load_dotenv()

# ─────────────────────────── CONFIG ───────────────────────────
PDF_DIR  = Path("pdfs")
IMG_DIR  = Path("/tmp/pdf_imgs")
DEVA_DIR = Path("OCR_deva")
IAST_DIR = Path("OCR_iast")
NUM_WORKERS = 8
TIMEOUT = 60

# Global variables for progress tracking and log capture
current_progress = {"percent": 0, "message": "", "logs": [], "process": ""}
progress_lock = threading.Lock()

def update_progress(percent, message, process=""):
    """Update progress and add log message"""
    with progress_lock:
        current_progress["percent"] = percent
        current_progress["message"] = message
        if process:
            current_progress["process"] = f"{process} ({percent}%)"
        current_progress["logs"].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # Keep only last 100 log entries
        if len(current_progress["logs"]) > 100:
            current_progress["logs"] = current_progress["logs"][-100:]

def get_progress():
    """Get current progress status"""
    with progress_lock:
        return current_progress.copy()

def reset_progress():
    """Reset progress for new operation"""
    with progress_lock:
        current_progress["percent"] = 0
        current_progress["message"] = ""
        current_progress["logs"] = []
        current_progress["process"] = ""

class GCVOCRWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id

        # Create Google Vision client (uses GOOGLE_APPLICATION_CREDENTIALS env var)
        self.client = vision.ImageAnnotatorClient()

    def grab_ocr(self, img_path: Path, process: str) -> str:
        """Upload one image and return recognised text using Google Cloud Vision."""

        text = ""
        try:
            with io.open(img_path, 'rb') as image_file:
                content = image_file.read()

            image = vision.Image(content=content)

            # Perform OCR
            response = self.client.text_detection(image=image)
            texts = response.text_annotations

            if texts:
                text = texts[0].description.strip()
                update_progress(current_progress["percent"], f"Worker successfully OCRed {img_path.name}", process)
            else:
                update_progress(current_progress["percent"], f"⚠️ Worker {self.worker_id}: No text found in {img_path.name}", process)

            if response.error.message:
                update_progress(current_progress["percent"], f"❌ Worker {self.worker_id}: API error - {response.error.message}", process)

        except Exception as e:
            update_progress(current_progress["percent"], f"⚠️ Worker {self.worker_id}: Error processing {img_path.name}: {str(e)}", process)

        return text

    def process_images(self, image_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue):
        """Process images from the queue."""
        process = "Running OCR on images"

        while True:
            try:
                img_path: Path = image_queue.get_nowait()
            except queue.Empty:
                break

            update_progress(current_progress["percent"], f"Worker {self.worker_id}: Processing {img_path.name}", process)
            recognised = self.grab_ocr(img_path, process)

            results_queue.put({
                'img_path': img_path,
                'text': recognised,
                'worker_id': self.worker_id
            })
            update_progress(current_progress["percent"], f"Worker {self.worker_id}: Completed {img_path.name} ({len(recognised)} chars)", process)

def pdftoppm_progress(current: int, total: int, process: str):
    # Guard against divide-by-zero and overshoot
    total = max(1, total)
    current = max(0, min(current, total))
    percent = int((current / total) * 100)
    update_progress(percent, f"Converting PDF page to PNG image: {current}/{total}", process)

def pdftoppm(pdf_path: Path, ppm: str, img_dir: Path) -> int:
    """Convert PDF to images with pdftoppm.
    Returns total page count on success, 0 on failure. Emits UI-friendly logs/progress.
    """
    process = "Converting PDF to PNG images"

    try:
        # Ensure temp dir is fresh for each PDF
        if img_dir.exists():
            shutil.rmtree(img_dir)
        img_dir.mkdir(parents=True, exist_ok=True)

        output_prefix = img_dir / "page"

        # Get total page count using pdfinfo
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True, text=True, check=True
        )

        pages = 0
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                try:
                    pages = int(line.split()[1])
                except Exception:
                    pages = 0
                break
        if not pages:
            update_progress(0, "Could not determine number of pages.", process)
            return 0

        # Start pdftoppm in background
        proc = subprocess.Popen(
            ["pdftoppm", f"-{ppm}", str(pdf_path), str(output_prefix)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True
        )

        # Initial zero update so UI shows the phase started
        pdftoppm_progress(0, pages, process)

        # Monitor progress
        while proc.poll() is None:
            try:
                count = sum(
                    1 for f in os.listdir(img_dir)
                    if f.startswith(str("page")) and f.endswith(f".{ppm}")
                )
            except FileNotFoundError:
                count = 0
            pdftoppm_progress(count, pages, process)
            time.sleep(1)

        # Read any stderr from pdftoppm
        if proc.returncode != 0:
            err = proc.stderr.read() if proc.stderr else ""
            update_progress(0, f"pdftoppm failed with code {proc.returncode}: {err.strip()}", process)
            return 0

        # Final update once finished
        try:
            count = sum(
                1 for f in os.listdir(img_dir)
                if f.startswith(str("page")) and f.endswith(f".{ppm}")
            )
        except FileNotFoundError:
            count = 0
        pdftoppm_progress(count, pages, process)
        update_progress(100, f"✅ Converted {pdf_path} → {ppm.upper()} images", process)
        return pages

    except Exception as e:
        update_progress(0, f"❌ Unexpected error during PDF conversion: {e}", process)
        return 0

def save_results(results_queue: multiprocessing.Queue, total_images: int, out_dir: Path, process: str):
    """Save OCR results to files (skip empty)."""
    saved_count = 0
    while saved_count < total_images:
        try:
            result = results_queue.get(timeout=10)
            img_path: Path = result['img_path']
            text: str = result['text']
            worker_id = result['worker_id']

            if text.strip():  # only save if non-empty
                out_file = out_dir / (img_path.stem + ".txt")
                out_file.write_text(text, encoding="utf-8")
                percent = int(((saved_count + 1) / total_images) * 100)
                update_progress(percent, f"✏️ Saved {out_file.name} (Worker {worker_id}) - {saved_count+1}/{total_images}", process)
            else:
                percent = int(((saved_count + 1) / total_images) * 100)
                update_progress(percent, f"⚠️ Skipped saving empty OCR for {img_path.name} (Worker {worker_id})", process)

            saved_count += 1

        except queue.Empty:
            update_progress(current_progress["percent"], "⚠️ Timeout waiting for results", process)
            break

def run_ocr_pipeline(img_dir: Path, deva_dir: Path, num_workers: int) -> None:
    """
    Run the OCR pipeline on all .png images in img_dir,
    save results to deva_dir.
    """
    process = "Running OCR on images"
    if not img_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {img_dir}")

    deva_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(img_dir.glob("*.png"))
    if not images:
        update_progress(current_progress["percent"], f"❌ No .png files found in {img_dir}", process)
        return

    update_progress(0, f"Found {len(images)} images to process using {num_workers} workers", process)

    image_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()

    for img in images:
        image_queue.put(img)

    processes: list[multiprocessing.Process] = []
    for i in range(num_workers):
        worker = GCVOCRWorker(i + 1)
        p = multiprocessing.Process(target=worker.process_images, args=(image_queue, results_queue))
        processes.append(p)
        p.start()

    save_results(results_queue, len(images), deva_dir, process)

    for p in processes:
        p.join()

    update_progress(100, f"✅ Completed processing {len(images)} images!", process)

# ───────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path):
    """Run OCR on all images for a given PDF and save Devanagari results only."""
    reset_progress()

    pdf_name = pdf_path.stem
    update_progress(0, f"Processing {pdf_name}", "Generating OCR from PNG images")

    # Step 0: Create save folder for this PDF
    deva_save_dir = DEVA_DIR / pdf_name

    # Check if OCR results already exist
    if deva_save_dir.exists() and any(deva_save_dir.iterdir()):
        update_progress(100, f"✅ OCR results already exist for {pdf_name}", "❌ OCR generation aborted")
        return

    deva_save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert PDF to PNG images
    pages = pdftoppm(pdf_path, ppm='png', img_dir=IMG_DIR)
    if pages <= 0:
        # Error already logged by pdftoppm; finalize progress so UI polling stops
        update_progress(100, "Aborted PDF to image conversion", "❌ OCR generation aborted")
        return

    # Step 2: Run OCR pipeline to get Devanagari text
    run_ocr_pipeline(IMG_DIR, deva_save_dir, NUM_WORKERS)

    # Step 3: Cleanup
    update_progress(100, "Cleaning up temporary files...", "Generating OCR from PNG images")
    shutil.rmtree(IMG_DIR, ignore_errors=True)

    update_progress(100, "OCR processing completed!", "Generating OCR from PNG images")
    return

def convert_to_iast(pdf_path: Path):
    """Convert existing Devanagari OCR results to IAST."""
    proc_good = "⏩️ Converting raw OCR to IAST"
    proc_fail = "❌ IAST conversion aborted"
    proc_done = "✅ IAST conversion completed"

    pdf_stem = pdf_path.stem
    reset_progress()
    update_progress(0, f"⏩️ Converting to IAST: {pdf_stem}.", proc_good)

    # Step 0: Check if Devanagari results exist
    deva_save_dir = DEVA_DIR / pdf_stem
    iast_save_dir = IAST_DIR / pdf_stem

    if not (deva_save_dir.exists() and any(deva_save_dir.iterdir())):
        update_progress(100, f"❌ No Devanagari OCR results for {pdf_stem}; please run OCR first.", proc_fail)
        return

    # Check if IAST results already exist
    if iast_save_dir.exists() and any(iast_save_dir.iterdir()):
        update_progress(100, f"✅ IAST results already exist for {pdf_stem}.", proc_fail)
        return

    iast_save_dir.mkdir(parents=True, exist_ok=True)

    update_progress(0, "⏩️ Reading Devanagari OCR files...", proc_good)

    # Step 1: Collect OCRed texts and transliterate to IAST
    txt_files = sorted(deva_save_dir.glob("*.txt"))
    total_files = len(txt_files)

    for i, txt_file in enumerate(txt_files):
        iast_text = []
        raw = txt_file.read_text(encoding="utf-8")
        for line in raw.splitlines():
            if line.strip():  # Only process non-empty lines
                iast = transliterate(line, DEVANAGARI, IAST)
                iast_text.append(iast)

        if iast_text:
            # Save IAST version
            iast_out_path = iast_save_dir / txt_file.name
            iast_out_path.write_text("\n".join(iast_text), encoding="utf-8")
            percent = int(((i + 1) / total_files) * 100)
            update_progress(percent, f"✏️ Converted {txt_file.name} ({i+1}/{total_files})", proc_good)
        else:
            percent = int(((i + 1) / total_files) * 100)
            update_progress(percent, f"⚠️ No valid OCR output for {txt_file.stem}; skipping IAST conversion.", proc_good)

    update_progress(100, f"✅ Completed IAST conversion for {pdf_stem}!", proc_done)
    return

# ───────────────────────────── MAIN ─────────────────────────────

def main():
    for pdf_path in PDF_DIR.glob("*.pdf"):
        try:
            process_pdf(pdf_path)
        except Exception as e:
            print(f"❌ Error with {pdf_path}: {e}")
    print("\n✅ Job done\n")

if __name__ == "__main__":
    main()
