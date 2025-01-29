from flask import Flask, render_template, request, jsonify, send_file, url_for
import easyocr
import pytesseract
import cv2
import os
import pandas as pd
from werkzeug.utils import secure_filename
import numpy as np
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

reader = easyocr.Reader(['en'])
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_easyocr(image_path):
    result = reader.readtext(image_path)
    extracted_text = " ".join([text[1] for text in result])
    return extracted_text

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img

def extract_text_tesseract(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Error: Image preprocessing failed."
    return pytesseract.image_to_string(img)

def enhance_text_with_bert(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    
    # Get BERT embeddings
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Use the last hidden state for enhancement
    embeddings = outputs.last_hidden_state.mean(dim=1)
    
    # Here you can add additional processing based on BERT embeddings
    enhanced_text = {
        'original': text,
        'structured': {
            'date': extract_date(text),
            'amount': extract_amount(text),
            'invoice_number': extract_invoice_number(text)
        }
    }
    return enhanced_text

def extract_date(text):
    # Add your date extraction logic here
    return "Date extraction placeholder"

def extract_amount(text):
    # Add your amount extraction logic here
    return "Amount extraction placeholder"

def extract_invoice_number(text):
    # Add your invoice number extraction logic here
    return "Invoice number extraction placeholder"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image
            easyocr_text = extract_text_easyocr(filepath)
            tesseract_text = extract_text_tesseract(filepath)
            
            # Enhance with BERT
            enhanced_easyocr = enhance_text_with_bert(easyocr_text)
            enhanced_tesseract = enhance_text_with_bert(tesseract_text)
            
            # Save results to CSV
            data = {
                'EasyOCR': easyocr_text,
                'Tesseract': tesseract_text
            }
            df = pd.DataFrame(data.items(), columns=['Method', 'Extracted Text'])
            csv_filename = f'results_{filename}.csv'
            csv_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
            df.to_csv(csv_path, index=False)
            
            return jsonify({
                'success': True,
                'easyocr': enhanced_easyocr,
                'tesseract': enhanced_tesseract,
                'csv_url': url_for('download_file', filename=csv_filename)
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/download/<filename>')
def download_file(filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True)
