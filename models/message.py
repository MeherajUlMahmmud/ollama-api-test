from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)  # Use Text for longer messages
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    session_id = db.Column(db.String(36), nullable=True)  # For grouping conversation sessions
    created_by = db.Column(db.String(100), nullable=True)  # User ID or identifier
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Optional fields for metadata
    token_count = db.Column(db.Integer, nullable=True)
    model_used = db.Column(db.String(50), nullable=True)  # e.g., 'claude-sonnet-4'

    def __init__(self, content, role, session_id=None, created_by=None, token_count=None, model_used=None):
        self.content = content
        self.role = role
        self.session_id = session_id
        self.created_by = created_by
        self.token_count = token_count
        self.model_used = model_used

    def to_dict(self):
        return {
            'id': self.id,
            'content': self.content,
            'role': self.role,
            'session_id': self.session_id,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'token_count': self.token_count,
            'model_used': self.model_used
        }
