import subprocess, time, os, shutil, multiprocessing, queue, io, dotenv, threading, uuid, concurrent.futures
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

# Job-based progress tracking system with multiprocessing support
import multiprocessing
manager = multiprocessing.Manager()
job_progress = manager.dict()
progress_lock = manager.Lock()

def update_progress(job_id, percent, message, process="", status="running"):
    """Update progress and add log message for a specific job"""
    with progress_lock:
        if job_id not in job_progress:
            job_progress[job_id] = manager.dict({"percent": 0, "message": "", "logs": manager.list(), "process": "", "status": "running"})
        
        job_data = job_progress[job_id]
        job_data["percent"] = percent
        job_data["message"] = message
        job_data["status"] = status
        if process:
            job_data["process"] = f"{process} ({percent}%)"
        job_data["logs"].append(f"[{time.strftime('%H:%M:%S')}] {message}")
        # Keep only last 100 log entries
        if len(job_data["logs"]) > 100:
            job_data["logs"] = job_data["logs"][-100:]
        job_progress[job_id] = job_data
        print(f"DEBUG: Job progress updated: {dict(job_data)}")  # Debug logging

def get_progress(job_id):
    """Get current progress status for a specific job"""
    with progress_lock:
        if job_id in job_progress:
            job_data = job_progress[job_id]
            return {
                "percent": job_data["percent"], 
                "message": job_data["message"], 
                "logs": list(job_data["logs"]), 
                "process": job_data["process"],
                "status": job_data["status"]
            }
        return {"percent": 0, "message": "Job not found", "logs": [], "process": "", "status": "done"}

def reset_progress(job_id):
    """Reset progress for a specific job"""
    with progress_lock:
        job_progress[job_id] = manager.dict({"percent": 0, "message": "", "logs": manager.list(), "process": "", "status": "running"})

def cleanup_job_progress(job_id):
    """Clean up progress tracking for a completed job"""
    with progress_lock:
        if job_id in job_progress:
            job_data = job_progress[job_id]
            job_data["status"] = "done"
            job_progress[job_id] = job_data
        # The job will be removed from memory by a separate cleanup process if needed

class GCVOCRWorker:
    def __init__(self, worker_id: int):
        self.worker_id = worker_id

        # Create Google Vision client (uses GOOGLE_APPLICATION_CREDENTIALS env var)
        self.client = vision.ImageAnnotatorClient()

    def grab_ocr(self, img_path: Path, process: str, job_id: str) -> str:
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
                update_progress(job_id, get_progress(job_id)["percent"], f"Worker successfully OCRed {img_path.name}", process)
            else:
                update_progress(job_id, get_progress(job_id)["percent"], f"⚠️ Worker {self.worker_id}: No text found in {img_path.name}", process)

            if response.error.message:
                update_progress(job_id, get_progress(job_id)["percent"], f"❌ Worker {self.worker_id}: API error - {response.error.message}", process)

        except Exception as e:
            update_progress(job_id, get_progress(job_id)["percent"], f"⚠️ Worker {self.worker_id}: Error processing {img_path.name}: {str(e)}", process)

        return text

    def process_images(self, image_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue, job_id: str):
        """Process images from the queue."""
        process = "Running OCR on images"

        while True:
            try:
                img_path: Path = image_queue.get_nowait()
            except queue.Empty:
                break

            update_progress(job_id, get_progress(job_id)["percent"], f"Worker {self.worker_id}: Processing {img_path.name}", process)
            recognised = self.grab_ocr(img_path, process, job_id)

            results_queue.put({
                'img_path': img_path,
                'text': recognised,
                'worker_id': self.worker_id
            })
            update_progress(job_id, get_progress(job_id)["percent"], f"Worker {self.worker_id}: Completed {img_path.name} ({len(recognised)} chars)", process)

def pdftoppm_progress(job_id: str, current: int, total: int, process: str):
    # Guard against divide-by-zero and overshoot
    total = max(1, total)
    current = max(0, min(current, total))
    percent = int((current / total) * 100)
    update_progress(job_id, percent, f"Converting PDF page to PNG image: {current}/{total}", process)

def pdftoppm_chunk(args):
    """Convert a chunk of PDF pages to images - designed for multiprocessing"""
    pdf_path, ppm, chunk_dir, start_page, end_page, job_id = args
    
    try:
        chunk_dir.mkdir(parents=True, exist_ok=True)
        output_prefix = chunk_dir / "page"
        
        # Convert specific page range
        cmd = [
            "pdftoppm", f"-{ppm}", 
            "-f", str(start_page), "-l", str(end_page),
            str(pdf_path), str(output_prefix)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            update_progress(job_id, get_progress(job_id)["percent"], 
                           f"⚠️ Chunk {start_page}-{end_page} failed: {result.stderr.strip()}", 
                           "Converting PDF to PNG images")
            return 0
        
        # Count generated images
        count = sum(1 for f in os.listdir(chunk_dir) if f.endswith(f".{ppm}"))
        update_progress(job_id, get_progress(job_id)["percent"], 
                       f"✅ Processed pages {start_page}-{end_page} ({count} images)", 
                       "Converting PDF to PNG images")
        return count
        
    except Exception as e:
        update_progress(job_id, get_progress(job_id)["percent"], 
                       f"❌ Error in chunk {start_page}-{end_page}: {str(e)}", 
                       "Converting PDF to PNG images")
        return 0

def pdftoppm(pdf_path: Path, ppm: str, img_dir: Path, job_id: str) -> int:
    """Multi-core PDF to images conversion with page chunking.
    Returns total page count on success, 0 on failure.
    """
    process = "Converting PDF to PNG images"

    try:
        # Ensure temp dir is fresh for each PDF and job-specific
        job_img_dir = img_dir / job_id
        if job_img_dir.exists():
            shutil.rmtree(job_img_dir)
        job_img_dir.mkdir(parents=True, exist_ok=True)

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
            update_progress(job_id, 0, "Could not determine number of pages.", process)
            return 0

        update_progress(job_id, 0, f"Starting multi-core conversion of {pages} pages", process)

        # Calculate optimal chunk size (aim for 4-8 chunks)
        max_workers = min(NUM_WORKERS, os.cpu_count() or 4)
        chunk_size = max(1, pages // max_workers)
        
        # Create chunk arguments for parallel processing
        chunks = []
        for i in range(0, pages, chunk_size):
            start_page = i + 1
            end_page = min(i + chunk_size, pages)
            chunk_dir = job_img_dir / f"chunk_{start_page}_{end_page}"
            chunks.append((pdf_path, ppm, chunk_dir, start_page, end_page, job_id))

        update_progress(job_id, 5, f"Processing {len(chunks)} chunks using {max_workers} cores", process)

        # Process chunks in parallel
        total_converted = 0
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(pdftoppm_chunk, chunk) for chunk in chunks]
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                try:
                    chunk_count = future.result()
                    total_converted += chunk_count
                    progress = int(20 + (i / len(chunks)) * 70)  # 20-90% for chunk processing
                    update_progress(job_id, progress, 
                                   f"Completed chunk {i}/{len(chunks)} ({total_converted}/{pages} pages)", 
                                   process)
                except Exception as e:
                    update_progress(job_id, get_progress(job_id)["percent"], 
                                   f"⚠️ Chunk {i} failed: {str(e)}", process)

        # Consolidate all chunk outputs into job-specific final directory
        update_progress(job_id, 90, "Consolidating chunk outputs...", process)
        
        final_img_dir = img_dir / f"job_{job_id}"
        if final_img_dir.exists():
            shutil.rmtree(final_img_dir)
        final_img_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect and rename all images to canonical format
        all_images = []
        for chunk_dir in job_img_dir.iterdir():
            if chunk_dir.is_dir():
                for img_file in chunk_dir.glob(f"*.{ppm}"):
                    # Extract page number from filename
                    try:
                        page_num = int(img_file.stem.split('-')[-1])
                        all_images.append((page_num, img_file))
                    except (ValueError, IndexError):
                        continue
        
        # Sort by page number and rename to canonical format
        all_images.sort(key=lambda x: x[0])
        for i, (page_num, img_file) in enumerate(all_images, 1):
            canonical_name = f"page-{i:06d}.{ppm}"
            shutil.move(str(img_file), str(final_img_dir / canonical_name))
        
        # Clean up chunk directories
        shutil.rmtree(job_img_dir, ignore_errors=True)
        
        update_progress(job_id, 100, f"✅ Converted {pdf_path} → {pages} {ppm.upper()} images using {max_workers} cores", process)
        return pages

    except Exception as e:
        update_progress(job_id, 0, f"❌ Unexpected error during PDF conversion: {e}", process, status="done")
        return 0

def save_results(results_queue: multiprocessing.Queue, total_images: int, out_dir: Path, process: str, job_id: str):
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
                update_progress(job_id, percent, f"✏️ Saved {out_file.name} (Worker {worker_id}) - {saved_count+1}/{total_images}", process)
            else:
                percent = int(((saved_count + 1) / total_images) * 100)
                update_progress(job_id, percent, f"⚠️ Skipped saving empty OCR for {img_path.name} (Worker {worker_id})", process)

            saved_count += 1

        except queue.Empty:
            update_progress(job_id, get_progress(job_id)["percent"], "⚠️ Timeout waiting for results", process)
            break

def run_ocr_pipeline(img_dir: Path, deva_dir: Path, num_workers: int, job_id: str) -> None:
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
        update_progress(job_id, get_progress(job_id)["percent"], f"❌ No .png files found in {img_dir}", process, status="done")
        return

    update_progress(job_id, 0, f"Found {len(images)} images to process using {num_workers} workers", process)

    image_queue = multiprocessing.Queue()
    results_queue = multiprocessing.Queue()

    for img in images:
        image_queue.put(img)

    processes: list[multiprocessing.Process] = []
    for i in range(num_workers):
        worker = GCVOCRWorker(i + 1)
        p = multiprocessing.Process(target=worker.process_images, args=(image_queue, results_queue, job_id))
        processes.append(p)
        p.start()

    save_results(results_queue, len(images), deva_dir, process, job_id)

    for p in processes:
        p.join()

    update_progress(job_id, 100, f"✅ Completed processing {len(images)} images!", process)

# ───────────────────────────────────────────────────────────────

def process_pdf(pdf_path: Path, job_id: str = None):
    """Run OCR on all images for a given PDF and save Devanagari results only."""
    if job_id is None:
        job_id = str(uuid.uuid4())
    
    reset_progress(job_id)

    pdf_name = pdf_path.stem
    update_progress(job_id, 0, f"Processing {pdf_name}", "Generating OCR from PNG images")

    # Step 0: Create save folder for this PDF
    deva_save_dir = DEVA_DIR / pdf_name

    # Check if OCR results already exist
    if deva_save_dir.exists() and any(deva_save_dir.iterdir()):
        update_progress(job_id, 100, f"✅ OCR results already exist for {pdf_name}", "❌ OCR generation aborted", status="done")
        return

    deva_save_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Convert PDF to PNG images  
    pages = pdftoppm(pdf_path, ppm='png', img_dir=IMG_DIR, job_id=job_id)
    if pages <= 0:
        # Error already logged by pdftoppm; finalize progress so UI polling stops
        update_progress(job_id, 100, "Aborted PDF to image conversion", "❌ OCR generation aborted", status="done")
        return

    # Reset progress for the next stage
    update_progress(job_id, 0, "Starting OCR extraction", "Generating OCR from PNG images")

    # Step 2: Run OCR pipeline to get Devanagari text from job-specific directory
    job_img_dir = IMG_DIR / f"job_{job_id}"
    run_ocr_pipeline(job_img_dir, deva_save_dir, NUM_WORKERS, job_id)

    # Step 3: Cleanup
    update_progress(job_id, 100, "Cleaning up temporary files...", "Generating OCR from PNG images")
    job_img_dir = IMG_DIR / f"job_{job_id}"
    shutil.rmtree(job_img_dir, ignore_errors=True)

    update_progress(job_id, 100, "OCR processing completed!", "Generating OCR from PNG images", status="done")
    cleanup_job_progress(job_id)
    return job_id

def convert_to_iast(pdf_path: Path, job_id: str = None):
    """Convert existing Devanagari OCR results to IAST."""
    if job_id is None:
        job_id = str(uuid.uuid4())
        
    proc_good = "⏩️ Converting raw OCR to IAST"
    proc_fail = "❌ IAST conversion aborted"
    proc_done = "✅ IAST conversion completed"

    pdf_stem = pdf_path.stem
    reset_progress(job_id)
    update_progress(job_id, 0, f"⏩️ Converting to IAST: {pdf_stem}.", proc_good)

    # Step 0: Check if Devanagari results exist
    deva_save_dir = DEVA_DIR / pdf_stem
    iast_save_dir = IAST_DIR / pdf_stem

    if not (deva_save_dir.exists() and any(deva_save_dir.iterdir())):
        update_progress(job_id, 100, f"❌ No Devanagari OCR results for {pdf_stem}; please run OCR first.", proc_fail, status="done")
        return

    # Check if IAST results already exist
    if iast_save_dir.exists() and any(iast_save_dir.iterdir()):
        update_progress(job_id, 100, f"✅ IAST results already exist for {pdf_stem}.", proc_fail, status="done")
        return

    iast_save_dir.mkdir(parents=True, exist_ok=True)

    update_progress(job_id, 0, "⏩️ Reading Devanagari OCR files...", proc_good)

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
            update_progress(job_id, percent, f"✏️ Converted {txt_file.name} ({i+1}/{total_files})", proc_good)
        else:
            percent = int(((i + 1) / total_files) * 100)
            update_progress(job_id, percent, f"⚠️ No valid OCR output for {txt_file.stem}; skipping IAST conversion.", proc_good)

    update_progress(job_id, 100, f"✅ Completed IAST conversion for {pdf_stem}!", proc_done, status="done")
    cleanup_job_progress(job_id)
    return job_id

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
