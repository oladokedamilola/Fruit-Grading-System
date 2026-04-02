# init_db.py
from webapp.app import app, db

with app.app_context():
    db.create_all()
    print("✓ Database created successfully at instance/fruit_grading.db")