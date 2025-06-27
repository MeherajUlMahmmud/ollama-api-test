import uuid

from flask import Flask, request
from flask_cors import CORS
from flask_migrate import Migrate

from config import Config
from logger import Logger, trace_id_context
from models.user import db
from routes.auth import auth_bp
from routes.liveness import liveness_bp
from routes.nid import nid_bp
from routes.query import query_bp

logger = Logger.get_logger()

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)
migrate = Migrate(app, db)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Register blueprints
app.register_blueprint(auth_bp, url_prefix='/api/auth')
app.register_blueprint(nid_bp, url_prefix='/api')
app.register_blueprint(liveness_bp, url_prefix='/api')
app.register_blueprint(query_bp, url_prefix='/api')

with app.app_context():
    db.create_all()


@app.before_request
def before_request():
    trace_id = str(uuid.uuid4())
    trace_id_context.set(trace_id)
    request.environ["trace_id"] = trace_id
    logger.info(f"Starting request with trace_id: {trace_id}")


@app.after_request
def after_request(response):
    logger.info(f"Request completed with trace_id: {trace_id_context.get()}")
    trace_id_context.set(None)
    return response


if __name__ == '__main__':
    logger.info("Starting Flask application")
    app.run(host='0.0.0.0', port=8000, debug=app.config['DEBUG'])
