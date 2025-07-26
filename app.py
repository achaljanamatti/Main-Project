import os
import logging
import shutil
from datetime import datetime
from flask import Flask, render_template, redirect, url_for, flash, request, session, g
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from flask_login import LoginManager, current_user, login_user, logout_user, login_required

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    pass

# Initialize the database
db = SQLAlchemy(model_class=Base)

# Create the Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-dev-secret-key")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

# Configure SQLite database
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///leukemia_detection.db"
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Configure file uploads
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["ALLOWED_EXTENSIONS"] = {"png", "jpg", "jpeg", "tif", "tiff"}
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB limit

# Ensure upload directory exists
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize the app with the extension
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in to access this page.'

from models import User, Analysis
from forms import LoginForm, RegistrationForm, UploadForm
from utils import allowed_file
from model_utils import preprocess_image, load_model, make_prediction

# Initialize white blood cancer detection model at startup
model = None

def initialize_model():
    global model
    try:
        model = load_model()
        logger.info("White blood cancer detection model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load the white blood cancer detection model: {str(e)}")
        model = None

# Add current year to all templates
@app.context_processor
def inject_current_year():
    return {'current_year': datetime.now().year}

# Initialize model within application context
with app.app_context():
    initialize_model()

@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))

@app.route('/')
def index():
    return redirect(url_for('dashboard'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password', 'danger')
            return redirect(url_for('login'))
        
        login_user(user, remember=form.remember_me.data)
        next_page = request.args.get('next')
        if not next_page or not next_page.startswith('/'):
            next_page = url_for('dashboard')
        return redirect(next_page)
    
    return render_template('login.html', title='Sign In', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Congratulations, you are now registered!', 'success')
        return redirect(url_for('login'))
    
    return render_template('register.html', title='Register', form=form)

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent analyses for the user
    recent_analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).limit(5)
    return render_template('dashboard.html', title='Dashboard', recent_analyses=recent_analyses)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        if 'image' not in request.files:
            flash('No file part', 'danger')
            return redirect(request.url)
        
        file = request.files['image']
        if file.filename == '':
            flash('No selected file', 'danger')
            return redirect(request.url)
        
        if file and allowed_file(file.filename, app.config["ALLOWED_EXTENSIONS"]):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the image and make prediction
            try:
                if model is None:
                    flash('Model not loaded. Please try again later.', 'danger')
                    return redirect(url_for('upload'))
                
                processed_image = preprocess_image(filepath)
                prediction, confidence = make_prediction(model, processed_image)
                
                # Save analysis to database
                analysis = Analysis(
                    user_id=current_user.id,
                    image_path=filepath,
                    result=prediction,
                    confidence=float(confidence),
                    timestamp=datetime.now()
                )
                db.session.add(analysis)
                db.session.commit()
                
                # Redirect to result page
                return redirect(url_for('result', analysis_id=analysis.id))
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                flash(f'Error processing image: {str(e)}', 'danger')
                return redirect(url_for('upload'))
        else:
            flash('File type not allowed. Please upload a valid image (png, jpg, jpeg, tif, tiff).', 'danger')
            return redirect(request.url)
    
    return render_template('upload.html', title='Upload Image', form=form)

@app.route('/result/<int:analysis_id>')
@login_required
def result(analysis_id):
    analysis = Analysis.query.get_or_404(analysis_id)
    
    # Ensure the analysis belongs to the current user
    if analysis.user_id != current_user.id:
        flash('You do not have permission to view this result.', 'danger')
        return redirect(url_for('dashboard'))
    
    return render_template('result.html', title='Analysis Result', analysis=analysis)

@app.route('/history')
@login_required
def history():
    page = request.args.get('page', 1, type=int)
    analyses = Analysis.query.filter_by(user_id=current_user.id).order_by(Analysis.timestamp.desc()).paginate(
        page=page, per_page=10, error_out=False)
    
    return render_template('history.html', title='Analysis History', analyses=analyses)

@app.route('/test-images')
@login_required
def test_images():
    """Display sample images for testing"""
    return render_template('test_images.html', title='Test Images')

@app.route('/bulk-test', methods=['GET', 'POST'])
@login_required
def bulk_test():
    """Display bulk testing interface"""
    # Count available test images
    leukemia_count = len([f for f in os.listdir('test_images/leukemia') if f.endswith('.png')])
    normal_count = len([f for f in os.listdir('test_images/normal') if f.endswith('.png')])
    
    # Initialize results variables
    results = None
    test_type = None
    correct_count = 0
    accuracy = 0
    avg_confidence = 0
    
    return render_template('bulk_test.html', title='Bulk Testing', 
                          leukemia_count=leukemia_count, normal_count=normal_count,
                          results=results, test_type=test_type,
                          correct_count=correct_count, accuracy=accuracy,
                          avg_confidence=avg_confidence)
                          
@app.route('/browse-test-data', methods=['GET'])
@login_required
def browse_test_data():
    """Browse test data images"""
    test_type = request.args.get('type', 'leukemia')
    page = request.args.get('page', 1, type=int)
    
    if test_type not in ['leukemia', 'normal']:
        test_type = 'leukemia'
    
    # Get image list
    test_dir = os.path.join('static', 'test_data', test_type)
    all_images = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    # Paginate images
    per_page = 20
    total_pages = (len(all_images) + per_page - 1) // per_page
    
    if page < 1:
        page = 1
    if page > total_pages:
        page = total_pages
    
    start_idx = (page - 1) * per_page
    end_idx = min(start_idx + per_page, len(all_images))
    current_images = all_images[start_idx:end_idx]
    
    # Create image URLs
    image_urls = [f'/static/test_data/{test_type}/{img}' for img in current_images]
    
    return render_template('browse_test_data.html', title='Browse Test Data',
                          test_type=test_type, image_urls=image_urls, 
                          current_page=page, total_pages=total_pages,
                          total_images=len(all_images))

@app.route('/run-bulk-test', methods=['POST'])
@login_required
def run_bulk_test():
    """Run bulk testing on multiple images"""
    if 'test_type' not in request.form or 'sample_count' not in request.form:
        flash('Invalid request parameters', 'danger')
        return redirect(url_for('bulk_test'))
    
    test_type = request.form['test_type']
    sample_count = int(request.form['sample_count'])
    
    if test_type not in ['leukemia', 'normal']:
        flash('Invalid test type', 'danger')
        return redirect(url_for('bulk_test'))
    
    # Check if model is loaded
    if model is None:
        flash('Model not loaded. Please try again later.', 'danger')
        return redirect(url_for('bulk_test'))
    
    # Get image list
    test_dir = os.path.join('test_images', test_type)
    image_files = [f for f in os.listdir(test_dir) if f.endswith('.png')]
    
    # Limit to requested sample count
    image_files = image_files[:sample_count]
    
    # Initialize results
    results = []
    correct_count = 0
    
    # Process each image
    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)
        
        try:
            # Process image and make prediction
            processed_image = preprocess_image(img_path)
            prediction, confidence = make_prediction(model, processed_image)
            
            # Determine if prediction is correct
            is_correct = (test_type == 'leukemia' and prediction == 'Leukemia') or \
                        (test_type == 'normal' and prediction == 'Non-Leukemia')
            
            if is_correct:
                correct_count += 1
            
            # Add to results
            results.append({
                'image': img_file,
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'is_correct': is_correct
            })
            
        except Exception as e:
            logger.error(f"Error processing test image {img_file}: {str(e)}")
            # Skip failed images
            continue
    
    # Calculate metrics
    accuracy = round((correct_count / len(results)) * 100, 2) if results else 0
    avg_confidence = round(sum(r['confidence'] for r in results) / len(results), 2) if results else 0
    
    # Count available test images for the form
    leukemia_count = len([f for f in os.listdir('test_images/leukemia') if f.endswith('.png')])
    normal_count = len([f for f in os.listdir('test_images/normal') if f.endswith('.png')])
    
    return render_template('bulk_test.html', title='Bulk Testing Results', 
                          leukemia_count=leukemia_count, normal_count=normal_count,
                          results=results, test_type=test_type,
                          correct_count=correct_count, accuracy=accuracy,
                          avg_confidence=avg_confidence)

@app.route('/run-upload-bulk-test', methods=['POST'])
@login_required
def run_upload_bulk_test():
    """Run bulk testing on multiple uploaded images"""
    if 'bulk_images' not in request.files or 'expected_result' not in request.form:
        flash('No files uploaded or missing parameters', 'danger')
        return redirect(url_for('bulk_test'))
    
    files = request.files.getlist('bulk_images')
    expected_result = request.form['expected_result']
    
    if not files or len(files) == 0:
        flash('No files selected', 'danger')
        return redirect(url_for('bulk_test'))
    
    # Check if model is loaded
    if model is None:
        flash('Model not loaded. Please try again later.', 'danger')
        return redirect(url_for('bulk_test'))
    
    # Initialize results
    results = []
    correct_count = 0
    test_type = expected_result  # For template display
    
    # Temporary directory for saving uploaded files
    temp_dir = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_bulk_test')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Process each uploaded file
        for file in files:
            if file and allowed_file(file.filename, {'png', 'jpg', 'jpeg', 'tif', 'tiff'}):
                # Save file temporarily
                filename = secure_filename(file.filename)
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                
                try:
                    # Process image and make prediction
                    processed_image = preprocess_image(file_path)
                    prediction, confidence = make_prediction(model, processed_image)
                    
                    # Determine if prediction is correct (if expected result is known)
                    is_correct = None
                    if expected_result != 'unknown':
                        is_correct = (expected_result == 'leukemia' and prediction == 'Leukemia') or \
                                    (expected_result == 'normal' and prediction == 'Non-Leukemia')
                        
                        if is_correct:
                            correct_count += 1
                    
                    # Add to results
                    results.append({
                        'image': filename,
                        'prediction': prediction,
                        'confidence': round(confidence * 100, 2),
                        'is_correct': is_correct
                    })
                    
                except Exception as e:
                    logger.error(f"Error processing uploaded image {filename}: {str(e)}")
                    # Add error to results
                    results.append({
                        'image': filename,
                        'prediction': 'Error',
                        'confidence': 0,
                        'is_correct': False,
                        'error': str(e)
                    })
        
        # Calculate metrics
        if expected_result != 'unknown' and results:
            accuracy = round((correct_count / len(results)) * 100, 2)
            avg_confidence = round(sum(r.get('confidence', 0) for r in results) / len(results), 2)
        else:
            accuracy = None
            avg_confidence = round(sum(r.get('confidence', 0) for r in results) / len(results), 2) if results else 0
        
    except Exception as e:
        logger.error(f"Error in bulk upload processing: {str(e)}")
        flash(f'Error processing uploaded images: {str(e)}', 'danger')
        return redirect(url_for('bulk_test'))
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
            except Exception as e:
                logger.error(f"Error removing temporary directory: {str(e)}")
    
    # Count available test images for the form (for the template)
    leukemia_count = len([f for f in os.listdir('test_images/leukemia') if f.endswith('.png')])
    normal_count = len([f for f in os.listdir('test_images/normal') if f.endswith('.png')])
    
    return render_template('bulk_test.html', title='Custom Bulk Testing Results', 
                          leukemia_count=leukemia_count, normal_count=normal_count,
                          results=results, test_type=test_type,
                          correct_count=correct_count, accuracy=accuracy,
                          avg_confidence=avg_confidence)

@app.route('/analyze-sample', methods=['POST'])
@login_required
def analyze_sample():
    """Analyze a sample image directly from the system"""
    if 'sample_path' not in request.form:
        flash('No sample selected', 'danger')
        return redirect(url_for('test_images'))
    
    sample_path = request.form['sample_path']
    
    try:
        # Ensure the path exists and is within our sample directories
        if not (sample_path.startswith('static/sample_images/positive/') or 
                sample_path.startswith('static/sample_images/negative/')):
            flash('Invalid sample path', 'danger')
            return redirect(url_for('test_images'))
        
        if not os.path.exists(sample_path):
            flash('Sample file not found', 'danger')
            return redirect(url_for('test_images'))
        
        # Process the image and make prediction
        if model is None:
            flash('Model not loaded. Please try again later.', 'danger')
            return redirect(url_for('test_images'))
        
        processed_image = preprocess_image(sample_path)
        prediction, confidence = make_prediction(model, processed_image)
        
        # Create a timestamp-based filename in uploads folder
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"{timestamp}_sample_{os.path.basename(sample_path)}"
        target_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Copy the sample file to uploads folder
        shutil.copy(sample_path, target_path)
        
        # Save analysis to database
        analysis = Analysis(
            user_id=current_user.id,
            image_path=target_path,
            result=prediction,
            confidence=float(confidence),
            notes="Sample image analysis",
            timestamp=datetime.now()
        )
        db.session.add(analysis)
        db.session.commit()
        
        # Redirect to result page
        return redirect(url_for('result', analysis_id=analysis.id))
    
    except Exception as e:
        logger.error(f"Error processing sample image: {str(e)}")
        flash(f'Error processing sample image: {str(e)}', 'danger')
        return redirect(url_for('test_images'))

# Create database tables
with app.app_context():
    db.create_all()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
