from flask import Flask, render_template, request, redirect, url_for, send_file
import os
from dotenv import load_dotenv
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import arabic_reshaper
from bidi.algorithm import get_display
from io import BytesIO
import time
import urllib.parse

app = Flask(__name__)

# Set base directory to the project root
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'Uploads')
CONTENT_FOLDER = os.path.join(BASE_DIR, 'content_text')
GEMINI_FOLDER = os.path.join(BASE_DIR, 'gemini_pdfs')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['CONTENT_FOLDER'] = CONTENT_FOLDER
app.config['GEMINI_FOLDER'] = GEMINI_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 100 MB limit

# Ensure folders exist
for folder in [UPLOAD_FOLDER, CONTENT_FOLDER, GEMINI_FOLDER]:
    try:
        os.makedirs(folder, exist_ok=True)
    except Exception as e:
        print(f"Error creating directory {folder}: {e}")

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Register fonts for different languages
try:
    FONTS_DIR = os.path.join(BASE_DIR, 'fonts')
    os.makedirs(FONTS_DIR, exist_ok=True)
    pdfmetrics.registerFont(TTFont('Amiri', os.path.join(FONTS_DIR, 'Amiri-Regular.ttf')))
    pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(FONTS_DIR, 'DejaVuSans.ttf')))
except Exception as e:
    print(f"Error registering fonts: {e}")

def has_extractable_text(pdf_file):
    try:
        with open(pdf_file, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages[:1]:
                text = page.extract_text() or ''
                if len(text.strip()) > 10:
                    return True
    except Exception as e:
        print(f"Error checking extractable text for {pdf_file}: {e}")
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
    except Exception as e:
        print(f"Error extracting text from {pdf_file}: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error extracting text: {e}\n")

def ocr_content(pdf_file, output_path, start_page, end_page, language='eng'):
    try:
        images = convert_from_path(pdf_file, dpi=300, first_page=start_page+1, last_page=end_page)
        mode = 'w' if start_page == 0 else 'a'
        with open(output_path, mode, encoding='utf-8') as f:
            for i, image in enumerate(images, start=start_page):
                text = pytesseract.image_to_string(image, lang=language)
                f.write(f"\n--- Page {i + 1} ---\n")
                f.write(text)
                f.write("\n")
    except Exception as e:
        print(f"Error performing OCR on {pdf_file}: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error performing OCR: {e}\n")

def generate_pdf_from_text(text, output_path, language='english'):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        if language.lower() in ['arabic']:
            font_name = 'Amiri'
            text = arabic_reshaper.reshape(text)
            text = get_display(text)
            style = ParagraphStyle(
                name='Arabic',
                fontName=font_name,
                fontSize=12,
                leading=14,
                alignment=2,
                spaceAfter=12,
                textColor=colors.black
            )
        else:
            font_name = 'DejaVuSans'
            style = ParagraphStyle(
                name='Normal',
                fontName=font_name,
                fontSize=12,
                leading=14,
                alignment=0,
                spaceAfter=12,
                textColor=colors.black
            )
        story = []
        lines = text.split('\n')
        for line in lines:
            if line.strip():
                story.append(Paragraph(line[:500], style))
                story.append(Spacer(1, 6))
        doc.build(story)
        buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buffer.read())
        time.sleep(0.1)
    except Exception as e:
        print(f"Error generating PDF for {language}: {e}")
        raise

def construct_prompt(content, grade, course, section, country, language):
    return f"""
Role:
You are an expert Instructional Content Designer specializing in creating visually organized, curriculum-aligned slide presentations for classroom use.

Objective:
Analyze the educational material provided (e.g., textbook chapter, article, teacher notes) and convert it into a well-structured, age-appropriate slide presentation for:
Grade: {grade}
Course: {course}
Curriculum Section: {section}
Country: {country}
Language: {language}

Slide Generation Instructions:
Content Structure & Fidelity:
Summarize and sequence content logically, following the original material closely.
Include all key ideas, definitions, explanations, examples, and terms.
Use as many slides as needed for clarity—no limit.
There is no maximum number of slides per lesson or sub-lesson—use however many are required to ensure full coverage and clear understanding.
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
Numbering & Labeling Instructions (Preventing Number Drift):
Maintain the original chapter and lesson numbering from the source content.
Do not continue lesson numbers across chapters.
Always label slides clearly using this structure:
Chapter X – Lesson X.Y – Slide Z
Reset the slide counter for each new lesson.
Preserve all source-based numbering exactly (e.g., 1.1, 1.2... 2.1, 2.2, etc.).

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
Continue parsing and converting content until the entire input is processed.
Keep format and tone consistent across all generated sets.
Assume that additional content will follow unless instructed otherwise.

Content:
{content}
"""

@app.route('/')
def index():
    txt_files = [f for f in os.listdir(app.config['CONTENT_FOLDER']) if f.endswith('.txt')]
    pdf_files = [f for f in os.listdir(app.config['GEMINI_FOLDER']) if f.endswith('.pdf')]
    return render_template('index.html', txt_files=txt_files, pdf_files=pdf_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files or not all(key in request.form for key in ['grade', 'course', 'section', 'language', 'country']):
        return redirect(url_for('index'))
    file = request.files['file']
    grade = request.form['grade']
    course = request.form['course']
    section = request.form['section']
    language = request.form['language']
    country = request.form['country']
    if file.filename == '' or not file.filename.endswith('.pdf'):
        return redirect(url_for('index'))

    base_filename = f"{course}_{grade}_{section}_{language}_{country}"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    txt_path = os.path.join(app.config['CONTENT_FOLDER'], f"{base_filename}.txt")
    
    file.save(pdf_path)
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
        
        # Check extractability once before processing
        is_extractable = has_extractable_text(pdf_path)
        
        # Process 3 pages at a time
        for start_page in range(0, num_pages, 3):
            end_page = min(start_page + 3, num_pages)
            if is_extractable:
                extract_text(pdf_path, txt_path, start_page, end_page)
            else:
                ocr_content(pdf_path, txt_path, start_page, end_page, language=language)
        
        # Verify all pages are processed
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for page_num in range(1, num_pages + 1):
                if f"--- Page {page_num} ---" not in content:
                    print(f"Warning: Page {page_num} missing in output text file")
                    # Attempt to reprocess missing page
                    if is_extractable:
                        extract_text(pdf_path, txt_path, page_num - 1, page_num)
                    else:
                        ocr_content(pdf_path, txt_path, page_num - 1, page_num, language=language)
    
        os.remove(pdf_path)
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error processing PDF: {e}")
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
        return redirect(url_for('index'))

@app.route('/view/<filename>')
def view_file(filename):
    file_path = os.path.join(app.config['CONTENT_FOLDER'], filename)
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return render_template('view.html', content=content, filename=filename)
        print(f"Text file not found: {file_path}")
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error serving text file: {e}")
        return redirect(url_for('index'))

@app.route('/generate_slides', methods=['POST'])
def generate_slides():
    filename = request.form.get('filename')
    grade = request.form.get('grade')
    course = request.form.get('course')
    section = request.form.get('section')
    country = request.form.get('country')
    language = request.form.get('language')
    
    if not all([filename, grade, course, section, country, language]):
        print("Missing required form fields")
        return redirect(url_for('index'))
    
    file_path = os.path.join(app.config['CONTENT_FOLDER'], filename)
    if not os.path.exists(file_path):
        print(f"Error: Text file not found: {file_path}")
        return redirect(url_for('index'))
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = construct_prompt(content, grade, course, section, country, language)
        response = model.generate_content(prompt)
        
        base_filename = f"{course}_{grade}_{section}_{language}_{country}"
        pdf_path = os.path.join(app.config['GEMINI_FOLDER'], f"{base_filename}_slides.pdf")
        generate_pdf_from_text(response.text, pdf_path, language=language)
        
        if not os.path.exists(pdf_path):
            print(f"PDF not generated: {pdf_path}")
            return redirect(url_for('index'))
        
        encoded_filename = urllib.parse.quote(f"{base_filename}_slides.pdf")
        return redirect(url_for('view_pdf', filename=encoded_filename))
    except Exception as e:
        print(f"Error generating slides: {e}")
        return redirect(url_for('index'))

@app.route('/view_pdf/<filename>')
def view_pdf(filename):
    decoded_filename = urllib.parse.unquote(filename)
    file_path = os.path.join(app.config['GEMINI_FOLDER'], decoded_filename)
    try:
        if os.path.exists(file_path):
            return send_file(file_path, mimetype='application/pdf')
        print(f"PDF not found: {file_path}")
        return redirect(url_for('index'))
    except Exception as e:
        print(f"Error serving PDF: {e}")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))