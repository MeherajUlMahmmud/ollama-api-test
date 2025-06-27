from datetime import datetime

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Query(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    query_text = db.Column(db.Text, nullable=False)
    tool_used = db.Column(db.String(50), nullable=True)  # e.g., 'web_search', 'repl', 'artifacts'
    message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=False)
    response_message_id = db.Column(db.Integer, db.ForeignKey('message.id'), nullable=True)
    execution_time = db.Column(db.Float, nullable=True)  # Time taken to execute
    status = db.Column(db.String(20), default='pending')  # 'pending', 'completed', 'failed'
    error_message = db.Column(db.Text, nullable=True)
    created_by = db.Column(db.String(100), nullable=True)  # User ID or identifier
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    message = db.relationship('Message', foreign_keys=[message_id], backref='queries')
    response_message = db.relationship('Message', foreign_keys=[response_message_id])

    def __init__(self, query_text, message_id, tool_used=None, response_message_id=None,
                 execution_time=None, status='pending', error_message=None):
        self.query_text = query_text
        self.message_id = message_id
        self.tool_used = tool_used
        self.response_message_id = response_message_id
        self.execution_time = execution_time
        self.status = status
        self.error_message = error_message

    def to_dict(self):
        return {
            'id': self.id,
            'query_text': self.query_text,
            'tool_used': self.tool_used,
            'message_id': self.message_id,
            'response_message_id': self.response_message_id,
            'execution_time': self.execution_time,
            'status': self.status,
            'error_message': self.error_message,
            'created_at': self.created_at.isoformat()
        }
