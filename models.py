from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    analyses = db.relationship('Analysis', backref='user', lazy='dynamic')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f'<User {self.username}>'

class Analysis(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    result = db.Column(db.String(50), nullable=False)  # 'leukemia' or 'non-leukemia'
    confidence = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<Analysis {self.id}, Result: {self.result}>'
    
    @property
    def formatted_confidence(self):
        """Return the confidence as a percentage with 2 decimal places"""
        return f"{self.confidence * 100:.2f}%"

    @property
    def formatted_date(self):
        """Return a nicely formatted date string"""
        return self.timestamp.strftime("%B %d, %Y at %H:%M")
    
    @property
    def is_positive(self):
        """Return True if the result is positive for leukemia"""
        return self.result.lower() == 'leukemia'
