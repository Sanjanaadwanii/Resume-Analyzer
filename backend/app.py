# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from analyzer import compute_scores

# PDF text extraction
from PyPDF2 import PdfReader
import io

app = Flask(__name__)
CORS(app)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    resume_text = data.get('resume_text','')
    jd_text = data.get('jd_text','')
    result = compute_scores(resume_text, jd_text)
    return jsonify(result)

@app.route('/analyze_file', methods=['POST'])
def analyze_file():
    """
    Accepts multipart/form-data with fields:
      - file (pdf)
      - jd_text (string) OR jd_text omitted and passed as part of form data
    Returns JSON same structure as compute_scores.
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # read job description from form field (if provided)
    jd_text = request.form.get('jd_text', '')

    # Only handle PDF uploads (quick check)
    filename = file.filename.lower()
    if not filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400

    # Read bytes and extract text using PyPDF2
    try:
        file_bytes = file.read()
        reader = PdfReader(io.BytesIO(file_bytes))
        extracted_text = []
        for page in reader.pages:
            try:
                text = page.extract_text()
            except Exception:
                text = None
            if text:
                extracted_text.append(text)
        resume_text = "\n".join(extracted_text).strip()
    except Exception as e:
        return jsonify({'error': f'Failed to parse PDF: {str(e)}'}), 500

    if not resume_text:
        # PDF might be scanned image; mention OCR requirement
        return jsonify({
            'error': 'No text extracted from PDF. If this is a scanned PDF (images), OCR is required.'
        }), 400

    # Run analyzer
    result = compute_scores(resume_text, jd_text)
    result['filename'] = file.filename
    # optionally return a short excerpt
    result['resume_excerpt'] = resume_text[:800]
    return jsonify(result)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

