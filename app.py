from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import logging
from werkzeug.utils import secure_filename
from pipeline import process_asl_video

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('asl_recognition.log')
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Step 1: Save uploaded video
    if 'video' not in request.files:
        logger.error("No video file in request")
        return 'No video uploaded', 400
    file = request.files['video']
    if file.filename == '':
        logger.error("Empty filename submitted")
        return 'No selected file', 400
    
    filename = secure_filename(file.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(video_path)
    logger.info(f"Saved uploaded video to {video_path}")

    # Process the video through our enhanced pipeline
    logger.info(f"Starting ASL recognition pipeline for {video_path}")
    results = process_asl_video(video_path)
    
    # Log the results
    logger.info(f"Video processed in {results['timing']['total']:.2f}s")
    logger.info(f"Detected {len(results['clips'])} word segments")
    logger.info(f"Recognized words: {results['words']}")
    logger.info(f"Generated sentence: '{results['sentence']}'")    # Render the results page
    return render_template('result.html', 
                         sentence=results['sentence'], 
                         raw_words=results['words'], 
                         clips=results['clips'],
                         zip=zip)

if __name__ == '__main__':
    app.run(debug=True)
