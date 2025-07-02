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

def ocr_content(pdf_path, output_path, start_page, end_page, language='eng'):
    try:
        images = convert_from_path(pdf_path, dpi=150, first_page=start_page+1, last_page=end_page)
        mode = 'w' if start_page == 0 else 'a'
        with open(output_path, mode, encoding='utf-8') as f:
            for i, image in enumerate(images, start=start_page):
                text = pytesseract.image_to_string(image, lang=language)
                f.write(f"\n--- Page {i + 1} ---\n")
                f.write(text)
                f.write("\n")
                time.sleep(0.1)  # Add delay to reduce load
    except Exception as e:
        print(f"Error performing OCR: {e}")
        with open(output_path, 'a', encoding='utf-8') as f:
            f.write(f"Error performing OCR: {e}\n")

from reportlab.pdfbase.pdfmetrics import stringWidth

from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.enums import TA_RIGHT, TA_LEFT, TA_CENTER

def generate_pdf_from_text(text, output_path, language='english'):
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        
        story = []
        lines = text.split('\n')
        
        for line in lines:
            if not line.strip():
                story.append(Spacer(1, 6))
                continue
            
            # Detect if line contains Arabic characters
            has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in line)
            has_english = any(c.isascii() and c.isalpha() for c in line)
            
            if has_arabic:
                # Configure for Arabic text
                font_name = 'Amiri'
                try:
                    # Reshape Arabic text properly
                    reshaped_text = arabic_reshaper.reshape(line)
                    bidi_text = get_display(reshaped_text)
                except Exception as e:
                    print(f"Error reshaping Arabic text: {e}")
                    bidi_text = line
                
                # Create Arabic style with proper RTL alignment
                arabic_style = ParagraphStyle(
                    name='Arabic',
                    fontName=font_name,
                    fontSize=12,
                    leading=16,
                    alignment=TA_RIGHT,  # Right alignment for Arabic
                    spaceAfter=12,
                    textColor=colors.black,
                    allowWidows=1,
                    allowOrphans=1,
                    splitLongWords=False,
                    wordWrap='RTL'
                )
                
                # Handle mixed content (Arabic + English)
                if has_english and has_arabic:
                    # For mixed content, create separate paragraphs for better control
                    # Split by common separators that might indicate language switch
                    import re
                    # Split on common punctuation that separates languages
                    parts = re.split(r'([.!?:;،؛])', bidi_text)
                    
                    for part in parts:
                        if part.strip():
                            part_has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in part)
                            if part_has_arabic:
                                p = Paragraph(part.strip(), arabic_style)
                            else:
                                # Use English style for non-Arabic parts
                                english_style = ParagraphStyle(
                                    name='English',
                                    fontName='DejaVuSans',
                                    fontSize=12,
                                    leading=16,
                                    alignment=TA_LEFT,
                                    spaceAfter=12,
                                    textColor=colors.black
                                )
                                p = Paragraph(part.strip(), english_style)
                            story.append(p)
                else:
                    # Pure Arabic text
                    p = Paragraph(bidi_text, arabic_style)
                    story.append(p)
            else:
                # Configure for non-Arabic text (English, etc.)
                font_name = 'DejaVuSans'
                english_style = ParagraphStyle(
                    name='English',
                    fontName=font_name,
                    fontSize=12,
                    leading=16,
                    alignment=TA_LEFT,  # Left alignment for English
                    spaceAfter=12,
                    textColor=colors.black
                )
                p = Paragraph(line, english_style)
                story.append(p)
            
            story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buffer.read())
        time.sleep(0.1)
    except Exception as e:
        print(f"Error generating PDF for {language}: {e}")
        raise

# Alternative approach for better mixed language handling
def generate_pdf_from_text_advanced(text, output_path, language='english'):
    """
    Advanced version with better mixed language support
    """
    try:
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
        
        story = []
        lines = text.split('\n')
        
        # Define styles
        arabic_style = ParagraphStyle(
            name='Arabic',
            fontName='Amiri',
            fontSize=12,
            leading=16,
            alignment=TA_RIGHT,
            spaceAfter=12,
            textColor=colors.black,
            allowWidows=1,
            allowOrphans=1,
            splitLongWords=False
        )
        
        english_style = ParagraphStyle(
            name='English',
            fontName='DejaVuSans',
            fontSize=12,
            leading=16,
            alignment=TA_LEFT,
            spaceAfter=12,
            textColor=colors.black
        )
        
        for line in lines:
            if not line.strip():
                story.append(Spacer(1, 12))
                continue
            
            # Process line for mixed content
            processed_line = process_mixed_language_line(line)
            
            if processed_line['type'] == 'arabic':
                p = Paragraph(processed_line['text'], arabic_style)
                story.append(p)
            elif processed_line['type'] == 'english':
                p = Paragraph(processed_line['text'], english_style)
                story.append(p)
            elif processed_line['type'] == 'mixed':
                # Handle mixed content by creating separate paragraphs
                for segment in processed_line['segments']:
                    if segment['type'] == 'arabic':
                        p = Paragraph(segment['text'], arabic_style)
                    else:
                        p = Paragraph(segment['text'], english_style)
                    story.append(p)
                    story.append(Spacer(1, 3))
            
            story.append(Spacer(1, 6))
        
        doc.build(story)
        buffer.seek(0)
        with open(output_path, 'wb') as f:
            f.write(buffer.read())
            
    except Exception as e:
        print(f"Error generating advanced PDF: {e}")
        raise

def process_mixed_language_line(line):
    """
    Process a line that may contain mixed Arabic and English content
    """
    import re
    
    has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in line)
    has_english = any(c.isascii() and c.isalpha() for c in line)
    
    if has_arabic and not has_english:
        # Pure Arabic
        try:
            reshaped_text = arabic_reshaper.reshape(line)
            bidi_text = get_display(reshaped_text)
            return {'type': 'arabic', 'text': bidi_text}
        except:
            return {'type': 'arabic', 'text': line}
    elif has_english and not has_arabic:
        # Pure English
        return {'type': 'english', 'text': line}
    elif has_arabic and has_english:
        # Mixed content - split and process separately
        segments = []
        # Simple approach: split by spaces and process each word/phrase
        words = line.split()
        current_segment = []
        current_type = None
        
        for word in words:
            word_has_arabic = any(0x0600 <= ord(c) <= 0x06FF for c in word)
            word_type = 'arabic' if word_has_arabic else 'english'
            
            if current_type is None:
                current_type = word_type
                current_segment.append(word)
            elif current_type == word_type:
                current_segment.append(word)
            else:
                # Type changed, save current segment and start new one
                if current_segment:
                    text = ' '.join(current_segment)
                    if current_type == 'arabic':
                        try:
                            reshaped_text = arabic_reshaper.reshape(text)
                            text = get_display(reshaped_text)
                        except:
                            pass
                    segments.append({'type': current_type, 'text': text})
                
                current_segment = [word]
                current_type = word_type
        
        # Add final segment
        if current_segment:
            text = ' '.join(current_segment)
            if current_type == 'arabic':
                try:
                    reshaped_text = arabic_reshaper.reshape(text)
                    text = get_display(reshaped_text)
                except:
                    pass
            segments.append({'type': current_type, 'text': text})
        
        return {'type': 'mixed', 'segments': segments}
    else:
        # Fallback
        return {'type': 'english', 'text': line}

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
        print(f"Error reading page count: {e}")
        return 0

@app.route('/')
def index():
    txt_files = [f for f in os.listdir(app.config['CONTENT_FOLDER']) if f.endswith('.txt')]
    pdf_files = [f for f in os.listdir(app.config['GEMINI_FOLDER']) if f.endswith('.pdf')]
    return render_template('index.html', txt_files=txt_files, pdf_files=pdf_files)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file uploaded")
        return redirect(url_for('index'))
    
    file = request.files['file']
    grade = request.form.get('grade', '').strip()
    course = request.form.get('course', '').strip()
    section = request.form.get('section', '').strip()
    language = request.form.get('language', '').strip()
    country = request.form.get('country', '').strip()

    if file.filename == '' or not file.filename.endswith('.pdf'):
        print("Invalid file or no file selected")
        return redirect(url_for('index'))

    if not all([grade, course, section, language, country]):
        print("Missing required form fields in upload")
        return redirect(url_for('index'))

    original_filename = os.path.splitext(file.filename)[0]
    base_filename = f"{course}_{grade}_{section}_{language}_{country}_{original_filename}"
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    txt_path = os.path.join(app.config['CONTENT_FOLDER'], f"{base_filename}.txt")
    
    file.save(pdf_path)
    
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
        
        is_extractable = has_extractable_text(pdf_path)
        
        for start_page in range(0, num_pages, 3):
            end_page = min(start_page + 3, num_pages)
            if is_extractable:
                extract_text(pdf_path, txt_path, start_page, end_page)
            else:
                ocr_content(pdf_path, txt_path, start_page, end_page, language=language)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
            for page_num in range(1, num_pages + 1):
                if f"--- Page {page_num} ---" not in content:
                    print(f"Warning: Page {page_num} missing in output text file")
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
    filename = request.form.get('filename', '').strip()
    grade = request.form.get('grade', '').strip()
    course = request.form.get('course', '').strip()
    section = request.form.get('section', '').strip()
    country = request.form.get('country', '').strip()
    language = request.form.get('language', '').strip()
    original_filename = os.path.splitext(filename)[0]
    
    if not all([filename, grade, course, section, country, language]):
        print("Missing required form fields in generate_slides")
        return redirect(url_for('index'))
    
    txt_path = os.path.join(app.config['CONTENT_FOLDER'], filename)
    if not os.path.exists(txt_path):
        print(f"Text file not found: {txt_path}")
        return redirect(url_for('index'))
    
    # Define paths for intermediate and final files
    base_filename = f"{course}_{grade}_{section}_{language}_{country}_{original_filename}"
    output_txt_path = os.path.join(app.config['GEMINI_FOLDER'], f"{base_filename}_gemini_response.txt")
    pdf_path = os.path.join(app.config['GEMINI_FOLDER'], f"{base_filename}_gemini_response.pdf")
    
    try:
        total_pages = get_page_count(txt_path)
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = construct_prompt(grade, course, section, country, language)
        
        with open(txt_path, 'r', encoding='utf-8') as f:
            full_content = f.read()
        
        # Ensure output file is created with UTF-8 encoding
        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write("")  # Initialize empty file
        
        print(f"Processing {total_pages} pages for Gemini...")
        
        if total_pages <= 30:
            print("Processing all pages in single request...")
            response = model.generate_content(prompt + "\n\nContent:\n" + full_content)
            if response and response.text:
                with open(output_txt_path, 'a', encoding='utf-8') as f:
                    f.write(response.text + "\n")
            else:
                print("No response from Gemini or empty response")
                # Clean up empty text file before returning
                if os.path.exists(output_txt_path):
                    os.remove(output_txt_path)
                    print(f"Cleaned up empty text file: {output_txt_path}")
                return redirect(url_for('index'))
        else:
            print(f"Processing {total_pages} pages in batches of 20...")
            pages = re.split(r'--- Page \d+ ---', full_content)[1:]
            batch_count = 0
            for start_page in range(0, total_pages, 20):
                end_page = min(start_page + 20, total_pages)
                batch_count += 1
                print(f"Processing batch {batch_count}: pages {start_page + 1}-{end_page}")
                
                page_content = "".join(pages[start_page:end_page])
                response = model.generate_content(prompt + "\n\nContent:\n" + page_content)
                if response and response.text:
                    with open(output_txt_path, 'a', encoding='utf-8') as f:
                        f.write(response.text + "\n")
                else:
                    print(f"No response for pages {start_page + 1}-{end_page}")
                    continue
        
        # Verify output file content
        with open(output_txt_path, 'r', encoding='utf-8') as f:
            response_text = f.read()
        
        if not response_text.strip():
            print(f"Output file is empty: {output_txt_path}")
            # Clean up empty text file
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
                print(f"Cleaned up empty text file: {output_txt_path}")
            return redirect(url_for('index'))
        
        print("Generating PDF from Gemini response...")
        # Generate PDF
        generate_pdf_from_text(response_text, pdf_path, language=language)
        
        if not os.path.exists(pdf_path):
            print(f"PDF not generated: {pdf_path}")
            # Clean up text file if PDF generation failed
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
                print(f"Cleaned up text file after PDF generation failure: {output_txt_path}")
            return redirect(url_for('index'))
        
        # PDF generated successfully, now clean up the intermediate text file
        try:
            os.remove(output_txt_path)
            print(f"Successfully cleaned up intermediate text file: {output_txt_path}")
        except Exception as cleanup_error:
            print(f"Warning: Could not delete intermediate text file {output_txt_path}: {cleanup_error}")
            # Don't fail the entire process if cleanup fails
        
        print(f"Process completed successfully. PDF available at: {pdf_path}")
        encoded_filename = urllib.parse.quote(f"{base_filename}_gemini_response.pdf")
        return redirect(url_for('view_pdf', filename=encoded_filename))
        
    except Exception as e:
        print(f"Error generating slides: {e}")
        # Clean up any intermediate files on error
        try:
            if os.path.exists(output_txt_path):
                os.remove(output_txt_path)
                print(f"Cleaned up text file after error: {output_txt_path}")
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
                print(f"Cleaned up partial PDF file after error: {pdf_path}")
        except Exception as cleanup_error:
            print(f"Warning: Error during cleanup: {cleanup_error}")
        
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