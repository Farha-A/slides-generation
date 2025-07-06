from flask import Flask, render_template, request, redirect, url_for, send_file, jsonify
import os
from dotenv import load_dotenv
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display
from io import BytesIO
import time
import urllib.parse
import re
import concurrent.futures
import logging
import threading
import queue
import json
from datetime import datetime
import tempfile
import shutil
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Increased timeout and size limits for large files
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB limit
app.config['UPLOAD_TIMEOUT'] = 1800  # 30 minutes

# Set base directory to the project root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
CONTENT_FOLDER = os.path.join(BASE_DIR, 'content_text')
GEMINI_FOLDER = os.path.join(BASE_DIR, 'gemini_pdfs')
PROGRESS_FOLDER = os.path.join(BASE_DIR, 'progress')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONTENT_FOLDER'] = CONTENT_FOLDER
app.config['GEMINI_FOLDER'] = GEMINI_FOLDER
app.config['PROGRESS_FOLDER'] = PROGRESS_FOLDER

# Ensure folders exist
for folder in [UPLOAD_FOLDER, CONTENT_FOLDER, GEMINI_FOLDER, PROGRESS_FOLDER]:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {folder}: {e}")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Global progress tracking
processing_status = {}
status_lock = threading.Lock()

# Register fonts for different languages
try:
    FONTS_DIR = os.path.join(BASE_DIR, 'fonts')
    os.makedirs(FONTS_DIR, exist_ok=True)
    pdfmetrics.registerFont(TTFont('Amiri', os.path.join(FONTS_DIR, 'Amiri-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(FONTS_DIR, 'DejaVuSans.ttf')))
except Exception as e:
    logger.error(f"Error registering fonts: {e}")

def update_progress(job_id, stage, progress, message=""):
    """Update processing progress"""
    with status_lock:
        processing_status[job_id] = {
            'stage': stage,
            'progress': progress,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
    
    # Also save to file for persistence
    try:
        progress_file = os.path.join(app.config['PROGRESS_FOLDER'], f"{job_id}.json")
        with open(progress_file, 'w') as f:
            json.dump(processing_status[job_id], f)
    except Exception as e:
        logger.error(f"Error saving progress to file: {e}")

def get_progress(job_id):
    """Get processing progress"""
    with status_lock:
        if job_id in processing_status:
            return processing_status[job_id]
    
    # Try to load from file
    try:
        progress_file = os.path.join(app.config['PROGRESS_FOLDER'], f"{job_id}.json")
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading progress from file: {e}")
    
    return None

def cleanup_progress(job_id):
    """Clean up progress tracking"""
    with status_lock:
        if job_id in processing_status:
            del processing_status[job_id]
    
    try:
        progress_file = os.path.join(app.config['PROGRESS_FOLDER'], f"{job_id}.json")
        if os.path.exists(progress_file):
            os.remove(progress_file)
    except Exception as e:
        logger.error(f"Error cleaning up progress file: {e}")

def has_extractable_text(pdf_file):
    """Check if PDF has extractable text - optimized for large files"""
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            # Check only first few pages for efficiency
            pages_to_check = min(3, len(reader.pages))
            
            for i in range(pages_to_check):
                text = reader.pages[i].extract_text() or ''
                if len(text.strip()) > 10:
                    return True
    except Exception as e:
        logger.error(f"Error checking extractable text for {pdf_file}: {e}")
    return False

def extract_text_streaming(pdf_file, output_path, start_page, end_page, job_id=None):
    """Extract text with progress updates"""
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            mode = 'w' if start_page == 0 else 'a'
            
            with open(output_path, mode, encoding='utf-8') as f:
                total_pages = end_page - start_page
                
                for i in range(start_page, min(end_page, len(reader.pages))):
                    if job_id:
                        progress = ((i - start_page + 1) / total_pages) * 100
                        update_progress(job_id, 'extracting', progress, f"Extracting page {i + 1}")
                    
                    text = reader.pages[i].extract_text() or ''
                    f.write(f"\n--- Page {i + 1} ---\n")
                    f.write(text)
                    f.write("\n")
                    f.flush()  # Ensure immediate write
                    
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_file}: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error extracting text: {e}\n")

def ocr_content_streaming(pdf_path, output_path, start_page, end_page, language='eng', job_id=None):
    """OCR content with progress updates and memory optimization"""
    try:
        logger.info(f"Starting OCR for pages {start_page+1} to {end_page} (language: {language})")
        
        # Process in smaller batches to manage memory
        batch_size = 1  # Process one page at a time for large files
        mode = 'w' if start_page == 0 else 'a'
        
        with open(output_path, mode, encoding='utf-8') as f:
            total_pages = end_page - start_page
            
            for batch_start in range(start_page, end_page, batch_size):
                batch_end = min(batch_start + batch_size, end_page)
                
                try:
                    # Convert batch of pages
                    images = convert_from_path(
                        pdf_path,
                        dpi=100,
                        first_page=batch_start+1,
                        last_page=batch_end,
                        fmt='jpeg',
                        jpegopt={'quality': 80, 'progressive': True, 'optimize': True}
                    )
                    
                    for i, image in enumerate(images, start=batch_start):
                        if job_id:
                            progress = ((i - start_page + 1) / total_pages) * 100
                            update_progress(job_id, 'ocr', progress, f"OCR processing page {i + 1}")
                        
                        logger.info(f"Processing page {i + 1} with OCR...")
                        
                        # Optimize image size
                        max_dimension = 1800
                        if max(image.size) > max_dimension:
                            ratio = max_dimension / max(image.size)
                            new_size = tuple(int(dim * ratio) for dim in image.size)
                            image = image.resize(new_size, Image.Resampling.LANCZOS)
                        
                        image = image.convert('L')
                        
                        try:
                            custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,!?;:-()[]{}'
                            if language == 'ara':
                                custom_config = r'--oem 3 --psm 6'
                            
                            text = pytesseract.image_to_string(
                                image,
                                lang=language,
                                config=custom_config,
                                timeout=180  # Increased timeout
                            )
                            
                            f.write(f"\n--- Page {i + 1} ---\n")
                            f.write(text)
                            f.write("\n")
                            f.flush()
                            
                        except Exception as page_error:
                            logger.error(f"OCR error on page {i + 1}: {page_error}")
                            f.write(f"\n--- Page {i + 1} (OCR Error) ---\n")
                            f.write(f"Error processing page: {str(page_error)}\n")
                            f.write("\n")
                        
                        # Clean up image from memory
                        del image
                    
                    # Clean up batch images
                    del images
                    
                except Exception as batch_error:
                    logger.error(f"Error processing batch {batch_start+1}-{batch_end}: {batch_error}")
                    f.write(f"\n--- Pages {batch_start+1}-{batch_end} (Batch Error) ---\n")
                    f.write(f"Error: {str(batch_error)}\n")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.1)
                
        logger.info(f"OCR completed for pages {start_page+1} to {end_page}")
        
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error performing OCR on pages {start_page+1}-{end_page}: {e}\n")

def generate_pdf_from_text(text, output_path, language='english'):
    """Generate PDF from text - same as original but with better error handling"""
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        story = []
        lines = text.split('\n')
        
        arabic_style = ParagraphStyle(
            name='Arabic',
            fontName='Amiri',
            fontSize=12,
            leading=16,
            alignment=2,  # TA_RIGHT
            spaceAfter=12,
            textColor=colors.black,
            allowWidows=1,
            allowOrphans=1,
            splitLongWords=False,
            wordWrap='RTL'
        )
        
        english_style = ParagraphStyle(
            name='English',
            fontName='DejaVuSans',
            fontSize=12,
            leading=16,
            alignment=1,  # TA_LEFT
            spaceAfter=12,
            textColor=colors.black
        )
        
        for line in lines:
            if not line.strip():
                story.append(Spacer(1, 6))
                continue
            
            has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in line)
            has_english = any(c.isascii() and c.isalpha() for c in line)
            
            if has_arabic:
                try:
                    reshaped_text = arabic_reshaper.reshape(line)
                    bidi_text = get_display(reshaped_text)
                except Exception as e:
                    logger.error(f"Error reshaping Arabic text: {e}")
                    bidi_text = line
                
                if has_english:
                    parts = re.split(r'([.!?:;،؛])', bidi_text)
                    for part in parts:
                        if part.strip():
                            part_has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in part)
                            style = arabic_style if part_has_arabic else english_style
                            p = Paragraph(part.strip(), style)
                            story.append(p)
                else:
                    p = Paragraph(bidi_text, arabic_style)
                    story.append(p)
            else:
                p = Paragraph(line, english_style)
                story.append(p)
            
            story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buffer.read())
        
        # Clean up buffer
        buffer.close()
        
    except Exception as e:
        logger.error(f"Error generating PDF for {language}: {e}")
        raise

def construct_prompt(grade, course, section, country, language):
    """Same as original"""
    return f"""
Role:
You are an expert Instructional Content Designer specializing in creating visually organized, curriculum-aligned slide presentations for classroom use.

Objective:
Generate a well-structured, age-appropriate slide presentation for:
Grade: {grade}
Course: {course}
Curriculum Section: {section}
Country: {country}
Language: {language}

Slide Generation Instructions:
Content Structure:
Create content based on typical curriculum for {course}, {section} for Grade {grade} in {country}.
Include key ideas, definitions, explanations, examples, and terms.
Use as many slides as needed for clarity—no limit.
If content includes multiple lessons or sub-lessons, generate:
A separate slide set for each.
Reset slide numbers at the start of each.
Label each lesson/sub-lesson clearly.
Concatenate all outputs into one continuous document.

Each Slide Must Include:
Slide Title: A clear and concise heading for the slide's main idea.
Bullet Points: 3–5 simplified, student-friendly bullets summarizing key information.
Suggested Visual: Description of a diagram, image, illustration, or chart that supports understanding.
Optional Think Prompt: A short reflective or analytical question aligned to Bloom's Taxonomy.
Numbering & Labeling Instructions:
Use standard chapter and lesson numbering for {course}, {section} in {country}.
Do not continue lesson numbers across chapters.
Always label slides clearly using this structure:
Chapter X – Lesson X.Y – Slide Z
Reset the slide counter for each new lesson.
Preserve standard curriculum-based numbering exactly (e.g., 1.1, 1.2... 2.1, 2.2, etc.).

Slide Style:
Use {language} at a reading level appropriate for Grade {grade}.
Use clear, engaging, instructive language.
Follow the "one concept per slide" principle—avoid text overload.

Curriculum & Learning Progression Guidelines:
Curriculum Alignment:
Structure content in line with the learning objectives and flow of the {section} curriculum of {country}.
Integrate subject-specific terminology and grade-appropriate academic language.

Bloom's Taxonomy Integration:
Start with slides that promote Remembering and Understanding.
Progress into Applying and Analyzing.
If appropriate, conclude with tasks that support Evaluating or Creating (e.g., student reflection, real-world problem-solving).

Output Format:
Number each slide (Slide 1, Slide 2, etc.).
Restart numbering for each new lesson or sub-lesson.
Keep format and tone consistent across all generated sets.
"""

def get_page_count(txt_path):
    """Same as original"""
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        page_numbers = re.findall(r'--- Page (\d+) ---', content)
        return len(page_numbers) if page_numbers else 0
    except Exception as e:
        logger.error(f"Error reading page count: {e}")
        return 0

def process_chunk_with_progress(pdf_path, txt_path, start_page, end_page, is_extractable, language, job_id=None):
    """Process chunk with progress tracking"""
    try:
        if is_extractable:
            extract_text_streaming(pdf_path, txt_path, start_page, end_page, job_id)
        else:
            ocr_content_streaming(pdf_path, txt_path, start_page, end_page, language=language, job_id=job_id)
    except Exception as chunk_error:
        logger.error(f"Error processing chunk {start_page}-{end_page}: {chunk_error}")
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Pages {start_page + 1}-{end_page} (Processing Error) ---\n")
            f.write(f"Error: {str(chunk_error)}\n")

def process_pdf_background(pdf_path, txt_path, grade, course, section, language, country, original_filename, job_id):
    """Background processing for large PDFs"""
    try:
        update_progress(job_id, 'analyzing', 10, "Analyzing PDF structure...")
        
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
        
        logger.info(f"PDF has {num_pages} pages")
        is_extractable = has_extractable_text(pdf_path)
        logger.info(f"PDF has extractable text: {is_extractable}")
        
        update_progress(job_id, 'processing', 20, f"Processing {num_pages} pages...")
        
        # Adjust chunk size based on file size and processing type
        chunk_size = 1 if not is_extractable and num_pages > 100 else (2 if not is_extractable else 8)
        
        # Process sequentially for large files to avoid memory issues
        for start in range(0, num_pages, chunk_size):
            end = min(start + chunk_size, num_pages)
            
            progress = 20 + ((start / num_pages) * 60)  # 20% to 80% for processing
            update_progress(job_id, 'processing', progress, f"Processing pages {start+1}-{end}")
            
            process_chunk_with_progress(pdf_path, txt_path, start, end, is_extractable, language, job_id)
        
        update_progress(job_id, 'verifying', 85, "Verifying processed content...")
        
        # Verify all pages
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_pages = []
        for page_num in range(1, num_pages + 1):
            if f"--- Page {page_num} ---" not in content:
                missing_pages.append(page_num)
        
        if missing_pages:
            logger.warning(f"Reprocessing missing pages: {missing_pages}")
            update_progress(job_id, 'reprocessing', 90, f"Reprocessing {len(missing_pages)} missing pages...")
            
            for page_num in missing_pages:
                process_chunk_with_progress(pdf_path, txt_path, page_num - 1, page_num, is_extractable, language, job_id)
        
        update_progress(job_id, 'completed', 100, "Processing completed successfully!")
        
        # Clean up uploaded PDF
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            logger.info(f"Cleaned up uploaded PDF: {pdf_path}")
        
    except Exception as e:
        logger.error(f"Error in background processing: {e}")
        update_progress(job_id, 'error', 0, f"Error: {str(e)}")
        
        # Clean up on error
        for cleanup_path in [pdf_path, txt_path]:
            if os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                    logger.info(f"Cleaned up: {cleanup_path}")
                except:
                    pass

@app.route('/')
def index():
    txt_files = [f for f in os.listdir(app.config['CONTENT_FOLDER']) if f.endswith('.txt')]
    pdf_files = [f for f in os.listdir(app.config['GEMINI_FOLDER']) if f.endswith('.pdf')]
    return render_template('index.html', txt_files=txt_files, pdf_files=pdf_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    """Async upload handler for large files"""
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    grade = request.form.get('grade', '').strip()
    course = request.form.get('course', '').strip()
    section = request.form.get('section', '').strip()
    language = request.form.get('language', '').strip()
    country = request.form.get('country', '').strip()

    if file.filename == '' or not file.filename.endswith('.pdf'):
        logger.error("Invalid file or no file selected")
        return jsonify({'error': 'Invalid file or no file selected'}), 400

    if not all([grade, course, section, language, country]):
        logger.error("Missing required form fields in upload")
        return jsonify({'error': 'Missing required form fields'}), 400

    try:
        # Generate job ID and secure filename
        job_id = f"{int(time.time())}_{secure_filename(file.filename)}"
        original_filename = os.path.splitext(file.filename)[0]
        base_filename = f"{course}_{grade}_{section}_{language}_{country}_{original_filename}"
        
        pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}.pdf")
        txt_path = os.path.join(app.config['CONTENT_FOLDER'], f"{base_filename}.txt")
        
        logger.info(f"Saving uploaded file: {file.filename}")
        update_progress(job_id, 'uploading', 5, "Saving uploaded file...")
        
        # Save file
        file.save(pdf_path)
        
        # Start background processing
        thread = threading.Thread(
            target=process_pdf_background,
            args=(pdf_path, txt_path, grade, course, section, language, country, original_filename, job_id)
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'job_id': job_id,
            'message': 'File uploaded successfully. Processing started.'
        })
        
    except Exception as e:
        logger.error(f"Error in upload: {e}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/progress/<job_id>')
def check_progress(job_id):
    """Check processing progress"""
    progress = get_progress(job_id)
    if progress:
        return jsonify(progress)
    else:
        return jsonify({'error': 'Job not found'}), 404

@app.route('/view/<filename>')
def view_file(filename):
    """Same as original"""
    file_path = os.path.join(app.config['CONTENT_FOLDER'], filename)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return render_template('view.html', content=content, filename=filename)
        logger.error(f"Text file not found: {file_path}")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error serving text file: {e}")
        return redirect(url_for('index'))

@app.route('/generate_slides', methods=['POST'])
def generate_slides():
    """Generate slides - optimized for large content"""
    filename = request.form.get('filename', '').strip()
    grade = request.form.get('grade', '').strip()
    course = request.form.get('course', '').strip()
    section = request.form.get('section', '').strip()
    country = request.form.get('country', '').strip()
    language = request.form.get('language', '').strip()
    original_filename = os.path.splitext(filename)[0]
    
    if not all([filename, grade, course, section, country, language]):
        logger.error("Missing required form fields in generate_slides")
        return redirect(url_for('index'))
    
    txt_path = os.path.join(app.config['CONTENT_FOLDER'], filename)
    if not os.path.exists(txt_path):
        logger.error(f"Text file not found: {txt_path}")
        return redirect(url_for('index'))
    
    base_filename = f"{course}_{grade}_{section}_{language}_{country}_{original_filename}"
    output_txt_path = os.path.join(app.config['GEMINI_FOLDER'], f"{base_filename}_gemini_response.txt")
    pdf_path = os.path.join(app.config['GEMINI_FOLDER'], f"{base_filename}_gemini_response.pdf")
    
    try:
        total_pages = get_page_count(txt_path)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = construct_prompt(grade, course, section, country, language)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("")
        
        logger.info(f"Processing {total_pages} pages for Gemini...")
        
        # Process in smaller batches for large files
        if total_pages <= 20:
            response = model.generate_content(prompt + "\n\nContent:\n" + full_content)
            if response and response.text:
                with open(output_txt_path, 'a', encoding='utf-8') as f:
                    f.write(response.text + "\n")
            else:
                logger.error("No response from Gemini or empty response")
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                return redirect(url_for('index'))
        else:
            # Process in smaller batches for large files
            pages = re.split(r'--- Page \d+ ---', full_content)[1:]
            batch_size = 15  # Smaller batch size for large files
            batch_count = 0
            
            for start_page in range(0, total_pages, batch_size):
                end_page = min(start_page + batch_size, total_pages)
                batch_count += 1
                logger.info(f"Processing batch {batch_count}: pages {start_page + 1}-{end_page}")
                
                page_content = "".join(pages[start_page:end_page])
                
                # Add retry logic for API calls
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        response = model.generate_content(prompt + "\n\nContent:\n" + page_content)
                        if response and response.text:
                            with open(output_txt_path, 'a', encoding='utf-8') as f:
                                f.write(response.text + "\n")
                            break
                        else:
                            logger.warning(f"No response for pages {start_page + 1}-{end_page}, attempt {attempt + 1}")
                            if attempt < max_retries - 1:
                                time.sleep(2 ** attempt)  # Exponential backoff
                    except Exception as api_error:
                        logger.error(f"API error for pages {start_page + 1}-{end_page}, attempt {attempt + 1}: {api_error}")
                        if attempt < max_retries - 1:
                            time.sleep(2 ** attempt)
                        else:
                            logger.error(f"Failed to process pages {start_page + 1}-{end_page} after {max_retries} attempts")
                
                # Small delay between batches
                time.sleep(1)
        
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            response_text = f.read()
        
        if not response_text.strip():
            logger.error(f"Output file is empty: {output_txt_path}")
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
            return redirect(url_for('index'))
        
        logger.info("Generating PDF from Gemini response...")
        generate_pdf_from_text(response_text, pdf_path, language=language)
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not generated: {pdf_path}")
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
            return redirect(url_for('index'))
        
        try:
            os.remove(output_txt_path)
            logger.info(f"Successfully cleaned up intermediate text file: {output_txt_path}")
        except Exception as cleanup_error:
            logger.warning(f"Could not delete intermediate text file {output_txt_path}: {cleanup_error}")
        
        logger.info(f"Process completed successfully. PDF available at: {pdf_path}")
        encoded_filename = urllib.parse.quote(f"{base_filename}_gemini_response.pdf")
        return redirect(url_for('view_pdf', filename=encoded_filename))
        
    except Exception as e:
        logger.error(f"Error generating slides: {e}")
        try:
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
                logger.info(f"Cleaned up text file after error: {output_txt_path}")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                logger.info(f"Cleaned up partial PDF file after error: {pdf_path}")
        except Exception as cleanup_error:
            logger.warning(f"Error during cleanup: {cleanup_error}")
        return redirect(url_for('index'))

@app.route('/view_pdf/<filename>')
def view_pdf(filename):
    """Same as original"""
    decoded_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(app.config['GEMINI_FOLDER'], decoded_filename)
    try:
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='application/pdf')
        logger.error(f"PDF not found: {file_path}")
        return redirect(url_for('index'))
    except Exception as e:
        logger.error(f"Error serving PDF: {e}")
        return redirect(url_for('index'))

@app.route('/cancel_job/<job_id>', methods=['POST'])
def cancel_job(job_id):
    """Cancel a processing job"""
    try:
        update_progress(job_id, 'cancelled', 0, "Job cancelled by user")
        
        # Clean up any files associated with this job
        cleanup_patterns = [
            os.path.join(app.config['UPLOAD_FOLDER'], f"{job_id}*"),
            os.path.join(app.config['CONTENT_FOLDER'], f"*{job_id}*"),
        ]
        
        for pattern in cleanup_patterns:
            import glob
            for file_path in glob.glob(pattern):
                try:
                    os.remove(file_path)
                    logger.info(f"Cleaned up cancelled job file: {file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up file {file_path}: {cleanup_error}")
        
        return jsonify({'success': True, 'message': 'Job cancelled successfully'})
    except Exception as e:
        logger.error(f"Error cancelling job: {e}")
        return jsonify({'error': f'Failed to cancel job: {str(e)}'}), 500

@app.route('/cleanup_old_files')
def cleanup_old_files():
    """Clean up old files and progress data"""
    try:
        import glob
        from datetime import datetime, timedelta
        
        # Clean up files older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        cleaned_count = 0
        
        # Clean up progress files
        for progress_file in glob.glob(os.path.join(app.config['PROGRESS_FOLDER'], '*.json')):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(progress_file))
                if file_time < cutoff_time:
                    os.remove(progress_file)
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not clean up progress file {progress_file}: {e}")
        
        # Clean up upload folder
        for upload_file in glob.glob(os.path.join(app.config['UPLOAD_FOLDER'], '*')):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(upload_file))
                if file_time < cutoff_time:
                    os.remove(upload_file)
                    cleaned_count += 1
            except Exception as e:
                logger.warning(f"Could not clean up upload file {upload_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} old files")
        return jsonify({'success': True, 'cleaned_files': cleaned_count})
    
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        return jsonify({'error': f'Cleanup failed: {str(e)}'}), 500

# Add error handler for large file uploads
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 500MB.'}), 413

# Add timeout handler
@app.errorhandler(408)
def timeout_handler(e):
    return jsonify({'error': 'Request timeout. Please try again or use a smaller file.'}), 408

# Health check endpoint
@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_jobs': len(processing_status)
    })

if __name__ == "__main__":
    # Clean up any old files on startup
    try:
        import glob
        from datetime import datetime, timedelta
        
        cutoff_time = datetime.now() - timedelta(hours=1)
        
        # Clean up any stale upload files
        for upload_file in glob.glob(os.path.join(UPLOAD_FOLDER, '*')):
            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(upload_file))
                if file_time < cutoff_time:
                    os.remove(upload_file)
                    logger.info(f"Cleaned up stale upload file: {upload_file}")
            except:
                pass
    except Exception as e:
        logger.warning(f"Error during startup cleanup: {e}")
    
    # Run the app
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), threaded=True)