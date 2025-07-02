from flask import Flask, render_template, request, redirect, url_for, send_file
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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Set base directory to the project root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
CONTENT_FOLDER = os.path.join(BASE_DIR, 'content_text')
GEMINI_FOLDER = os.path.join(BASE_DIR, 'gemini_pdfs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONTENT_FOLDER'] = CONTENT_FOLDER
app.config['GEMINI_FOLDER'] = GEMINI_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MB limit

# Ensure folders exist
for folder in [UPLOAD_FOLDER, CONTENT_FOLDER, GEMINI_FOLDER]:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        logger.error(f"Error creating directory {folder}: {e}")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Register fonts for different languages
try:
    FONTS_DIR = os.path.join(BASE_DIR, 'fonts')
    os.makedirs(FONTS_DIR, exist_ok=True)
    pdfmetrics.registerFont(TTFont('Amiri', os.path.join(FONTS_DIR, 'Amiri-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(FONTS_DIR, 'DejaVuSans.ttf')))
except Exception as e:
    logger.error(f"Error registering fonts: {e}")

def has_extractable_text(pdf_file):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages[:1]:
                text = page.extract_text() or ''
                if len(text.strip()) > 10:
                    return True
    except Exception as e:
        logger.error(f"Error checking extractable text for {pdf_file}: {e}")
    return False

def extract_text(pdf_file, output_path, start_page, end_page):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            mode = 'w' if start_page == 0 else 'a'
            with open(output_path, mode, encoding='utf-8') as f:
                for i in range(start_page, min(end_page, len(reader.pages))):
                    text = reader.pages[i].extract_text() or ''
                    f.write(f"\n--- Page {i + 1} ---\n")
                    f.write(text)
                    f.write("\n")
                    f.flush()  # Ensure immediate write
    except Exception as e:
        logger.error(f"Error extracting text from {pdf_file}: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error extracting text: {e}\n")

def ocr_content(pdf_path, output_path, start_page, end_page, language='eng'):
    try:
        logger.info(f"Starting OCR for pages {start_page+1} to {end_page} (language: {language})")
        images = convert_from_path(
            pdf_path,
            dpi=100,
            first_page=start_page+1,
            last_page=end_page,
            fmt='jpeg',
            jpegopt={'quality': 85, 'progressive': True, 'optimize': True}
        )
        
        mode = 'w' if start_page == 0 else 'a'
        with open(output_path, mode, encoding='utf-8') as f:
            for i, image in enumerate(images, start=start_page):
                logger.info(f"Processing page {i + 1} with OCR...")
                max_dimension = 2000
                if max(image.size) > max_dimension:
                    ratio = max_dimension / max(image.size)
                    new_size = tuple(int(dim * ratio) for dim in image.size)
                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                
                image = image.convert('L')
                try:
                    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz .,!?;:-()[]{}"\''
                    if language == 'ara':
                        custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(
                        image,
                        lang=language,
                        config=custom_config,
                        timeout=120
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
                
                time.sleep(0.2)
                del image
                
        logger.info(f"OCR completed for pages {start_page+1} to {end_page}")
    except Exception as e:
        logger.error(f"Error performing OCR: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error performing OCR on pages {start_page+1}-{end_page}: {e}\n")

def generate_pdf_from_text(text, output_path, language='english'):
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
        time.sleep(0.1)
    except Exception as e:
        logger.error(f"Error generating PDF for {language}: {e}")
        raise

def construct_prompt(grade, course, section, country, language):
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
Slide Title: A clear and concise heading for the slide’s main idea.
Bullet Points: 3–5 simplified, student-friendly bullets summarizing key information.
Suggested Visual: Description of a diagram, image, illustration, or chart that supports understanding.
Optional Think Prompt: A short reflective or analytical question aligned to Bloom’s Taxonomy.
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
Follow the “one concept per slide” principle—avoid text overload.

Curriculum & Learning Progression Guidelines:
Curriculum Alignment:
Structure content in line with the learning objectives and flow of the {section} curriculum of {country}.
Integrate subject-specific terminology and grade-appropriate academic language.

Bloom’s Taxonomy Integration:
Start with slides that promote Remembering and Understanding.
Progress into Applying and Analyzing.
If appropriate, conclude with tasks that support Evaluating or Creating (e.g., student reflection, real-world problem-solving).

Output Format:
Number each slide (Slide 1, Slide 2, etc.).
Restart numbering for each new lesson or sub-lesson.
Keep format and tone consistent across all generated sets.
"""

def get_page_count(txt_path):
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        page_numbers = re.findall(r'--- Page (\d+) ---', content)
        return len(page_numbers) if page_numbers else 0
    except Exception as e:
        logger.error(f"Error reading page count: {e}")
        return 0

def process_chunk(pdf_path, txt_path, start_page, end_page, is_extractable, language):
    try:
        if is_extractable:
            extract_text(pdf_path, txt_path, start_page, end_page)
        else:
            ocr_content(pdf_path, txt_path, start_page, end_page, language=language)
    except Exception as chunk_error:
        logger.error(f"Error processing chunk {start_page}-{end_page}: {chunk_error}")
        with open(txt_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- Pages {start_page + 1}-{end_page} (Processing Error) ---\n")
            f.write(f"Error: {str(chunk_error)}\n")

@app.route('/')
def index():
    txt_files = [f for f in os.listdir(app.config['CONTENT_FOLDER']) if f.endswith('.txt')]
    pdf_files = [f for f in os.listdir(app.config['GEMINI_FOLDER']) if f.endswith('.pdf')]
    return render_template('index.html', txt_files=txt_files, pdf_files=pdf_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        logger.error("No file uploaded")
        return redirect(url_for('index'))
    
    file = request.files['file']
    grade = request.form.get('grade', '').strip()
    course = request.form.get('course', '').strip()
    section = request.form.get('section', '').strip()
    language = request.form.get('language', '').strip()
    country = request.form.get('country', '').strip()

    if file.filename == '' or not file.filename.endswith('.pdf'):
        logger.error("Invalid file or no file selected")
        return redirect(url_for('index'))

    if not all([grade, course, section, language, country]):
        logger.error("Missing required form fields in upload")
        return redirect(url_for('index'))

    original_filename = os.path.splitext(file.filename)[0]
    base_filename = f"{course}_{grade}_{section}_{language}_{country}_{original_filename}"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    txt_path = os.path.join(app.config['CONTENT_FOLDER'], f"{base_filename}.txt")
    
    try:
        logger.info(f"Saving uploaded file: {file.filename}")
        file.save(pdf_path)
        
        with open(pdf_path, 'rb') as pdf_file:
            reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(reader.pages)
        
        logger.info(f"PDF has {num_pages} pages")
        is_extractable = has_extractable_text(pdf_path)
        logger.info(f"PDF has extractable text: {is_extractable}")
        
        chunk_size = 2 if not is_extractable else 5
        chunks = [(pdf_path, txt_path, start, min(start + chunk_size, num_pages), is_extractable, language)
                  for start in range(0, num_pages, chunk_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(process_chunk, *chunk) for chunk in chunks]
            concurrent.futures.wait(futures, timeout=300)  # 5-minute timeout per chunk
        
        # Verify all pages
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        missing_pages = []
        for page_num in range(1, num_pages + 1):
            if f"--- Page {page_num} ---" not in content:
                missing_pages.append(page_num)
        
        if missing_pages:
            logger.warning(f"Reprocessing missing pages: {missing_pages}")
            for page_num in missing_pages:
                process_chunk(pdf_path, txt_path, page_num - 1, page_num, is_extractable, language)
        
        os.remove(pdf_path)
        logger.info(f"Cleaned up uploaded PDF: {pdf_path}")
        return redirect(url_for('index'))
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        for cleanup_path in [pdf_path, txt_path]:
            if os.path.exists(cleanup_path):
                try:
                    os.remove(cleanup_path)
                    logger.info(f"Cleaned up: {cleanup_path}")
                except:
                    pass
        return redirect(url_for('index'))

@app.route('/view/<filename>')
def view_file(filename):
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
        
        if total_pages <= 30:
            response = model.generate_content(prompt + "\n\nContent:\n" + full_content)
            if response and response.text:
                with open(output_txt_path, 'a', encoding='utf-8') as f:
                    f.write(response.text + "\n")
            else:
                logger.error("No response from Gemini or empty response")
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                    logger.info(f"Cleaned up empty text file: {output_txt_path}")
                return redirect(url_for('index'))
        else:
            pages = re.split(r'--- Page \d+ ---', full_content)[1:]
            batch_count = 0
            for start_page in range(0, total_pages, 20):
                end_page = min(start_page + 20, total_pages)
                batch_count += 1
                logger.info(f"Processing batch {batch_count}: pages {start_page + 1}-{end_page}")
                
                page_content = "".join(pages[start_page:end_page])
                response = model.generate_content(prompt + "\n\nContent:\n" + page_content)
                if response and response.text:
                    with open(output_txt_path, 'a', encoding='utf-8') as f:
                        f.write(response.text + "\n")
                else:
                    logger.warning(f"No response for pages {start_page + 1}-{end_page}")
                    continue
        
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            response_text = f.read()
        
        if not response_text.strip():
            logger.error(f"Output file is empty: {output_txt_path}")
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
                logger.info(f"Cleaned up empty text file: {output_txt_path}")
            return redirect(url_for('index'))
        
        logger.info("Generating PDF from Gemini response...")
        generate_pdf_from_text(response_text, pdf_path, language=language)
        
        if not os.path.exists(pdf_path):
            logger.error(f"PDF not generated: {pdf_path}")
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
                logger.info(f"Cleaned up text file after PDF generation failure: {output_txt_path}")
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))