"""
Verdana — Backend API
Flask + SQLite + JWT

Endpoints:
  Auth       POST /api/auth/signup
             POST /api/auth/login
             POST /api/auth/logout
             GET  /api/auth/me
             POST /api/auth/forgot-password

  Detection  POST /api/detect          (upload image + plant type → save result)
             GET  /api/detections       (user's history)
             GET  /api/detections/<id>  (single detection)
             DELETE /api/detections/<id>

  Contact    POST /api/contact
             GET  /api/contact          (admin: all messages)

  Newsletter POST /api/newsletter/subscribe
             GET  /api/newsletter/list   (admin)

  Profile    GET  /api/profile
             PUT  /api/profile
             PUT  /api/profile/password

  Admin      GET  /api/admin/users
             GET  /api/admin/stats
"""

import os, uuid, hashlib, hmac, secrets, json, re, io
from datetime import datetime, timedelta, timezone
from functools import wraps
from flask import Flask, request, jsonify, send_from_directory, g, render_template
from flask_cors import CORS
import sqlite3

# ─── ML Model (lazy-loaded on first use) ──────────────────────
_model      = None
_model_path = None   # set after BASE_DIR is defined

CLASS_NAMES = ['Healthy', 'Powdery', 'Rust']

DISEASE_RECOMMENDATIONS = {
    'Healthy': [
        'Plant looks healthy — keep up regular care!',
        'Water consistently and ensure good drainage.',
        'Monitor periodically for any early signs of stress.',
    ],
    'Powdery': [
        'Apply a fungicide spray (neem oil or sulfur-based) every 7–10 days.',
        'Improve air circulation around the plant.',
        'Avoid overhead watering; water at the base.',
        'Remove and dispose of heavily infected leaves.',
    ],
    'Rust': [
        'Remove and destroy infected leaves immediately.',
        'Apply copper-based or sulfur fungicide every 7–10 days.',
        'Avoid wetting foliage when watering.',
        'Ensure good spacing between plants for airflow.',
    ],
}

def get_model():
    """Lazy-load the Keras model on first inference request."""
    global _model
    if _model is None:
        try:
            import tensorflow as tf          # noqa: F401
            from tensorflow import keras
            if _model_path and os.path.exists(_model_path):
                _model = keras.models.load_model(_model_path)
            else:
                raise FileNotFoundError(f'Model file not found at: {_model_path}')
        except Exception as e:
            raise RuntimeError(f'Could not load model: {e}')
    return _model


def preprocess_image(file_obj):
    """Resize image to 224×224 and normalise to [0, 1]."""
    import numpy as np
    from PIL import Image
    img = Image.open(file_obj).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img, dtype='float32') / 255.0
    return arr[None, ...]   # shape (1, 224, 224, 3)


# ─── App setup ────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DB_PATH   = os.path.join(BASE_DIR, 'verdana.db')
UPLOAD_DIR = os.path.join(BASE_DIR, 'static', 'uploads')
SECRET_KEY = os.environ.get('SECRET_KEY', 'verdana-secret-change-in-prod-2026')

# Point to the trained model (place plant_disease_model.h5 next to app.py)
_model_path = os.path.join(BASE_DIR, 'plant_disease_model.h5')

app = Flask(__name__, static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024   # 10 MB
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ─── Simple JWT (no extra lib needed) ─────────────────────────
import base64, time as _time

def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode()

def _unb64(s: str) -> bytes:
    pad = 4 - len(s) % 4
    return base64.urlsafe_b64decode(s + '=' * pad)

def create_token(user_id: int, is_admin: bool = False, expires_hours: int = 72) -> str:
    payload = {
        'sub': user_id,
        'adm': is_admin,
        'exp': int(_time.time()) + expires_hours * 3600,
        'jti': secrets.token_hex(8)
    }
    header  = _b64(b'{"alg":"HS256","typ":"JWT"}')
    body    = _b64(json.dumps(payload).encode())
    sig     = _b64(hmac.new(SECRET_KEY.encode(), f'{header}.{body}'.encode(), hashlib.sha256).digest())
    return f'{header}.{body}.{sig}'

def verify_token(token: str):
    try:
        header, body, sig = token.split('.')
        expected = _b64(hmac.new(SECRET_KEY.encode(), f'{header}.{body}'.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(sig, expected):
            return None
        payload = json.loads(_unb64(body))
        if payload['exp'] < _time.time():
            return None
        return payload
    except Exception:
        return None

def require_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        auth = request.headers.get('Authorization', '')
        token = auth.removeprefix('Bearer ').strip()
        if not token:
            token = request.cookies.get('token', '')
        payload = verify_token(token)
        if not payload:
            return jsonify({'error': 'Unauthorized'}), 401
        g.user_id  = payload['sub']
        g.is_admin = payload.get('adm', False)
        return f(*args, **kwargs)
    return wrapper

def require_admin(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not g.is_admin:
            return jsonify({'error': 'Forbidden'}), 403
        return f(*args, **kwargs)
    return require_auth(wrapper)

# ─── Database ─────────────────────────────────────────────────
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA journal_mode=WAL')
        g.db.execute('PRAGMA foreign_keys=ON')
    return g.db

@app.teardown_appcontext
def close_db(exc):
    db = g.pop('db', None)
    if db:
        db.close()

def init_db():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.executescript('''
    CREATE TABLE IF NOT EXISTS users (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        first_name  TEXT NOT NULL,
        last_name   TEXT NOT NULL,
        email       TEXT UNIQUE NOT NULL,
        password    TEXT NOT NULL,         -- sha256(salt+password)
        salt        TEXT NOT NULL,
        user_type   TEXT DEFAULT 'gardener',
        phone       TEXT,
        is_admin    INTEGER DEFAULT 0,
        created_at  TEXT DEFAULT (datetime('now')),
        last_login  TEXT
    );

    CREATE TABLE IF NOT EXISTS detections (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        image_filename  TEXT,
        plant_type      TEXT,
        disease_name    TEXT,
        confidence      REAL,
        recommendations TEXT,   -- JSON array
        notes           TEXT,
        created_at      TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS contact_messages (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER REFERENCES users(id),
        name        TEXT NOT NULL,
        email       TEXT NOT NULL,
        subject     TEXT NOT NULL,
        message     TEXT NOT NULL,
        status      TEXT DEFAULT 'unread',
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS newsletter (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        email       TEXT UNIQUE NOT NULL,
        user_id     INTEGER REFERENCES users(id),
        subscribed  INTEGER DEFAULT 1,
        created_at  TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS password_resets (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        token       TEXT NOT NULL,
        expires_at  TEXT NOT NULL,
        used        INTEGER DEFAULT 0
    );
    ''')
    db.commit()
    db.close()
    print('✓ Database initialised')

# ─── Helpers ──────────────────────────────────────────────────
EMAIL_RE = re.compile(r'^[^\s@]+@[^\s@]+\.[^\s@]+$')

def hash_password(password: str, salt: str = None):
    if not salt:
        salt = secrets.token_hex(16)
    h = hashlib.sha256(f'{salt}{password}'.encode()).hexdigest()
    return h, salt

def row_to_dict(row):
    return dict(row) if row else None

def paginate(query_rows, page, per_page=20):
    page = max(1, int(page or 1))
    total = len(query_rows)
    start = (page - 1) * per_page
    items = query_rows[start:start + per_page]
    return {
        'items': [dict(r) for r in items],
        'total': total,
        'page': page,
        'pages': max(1, -(-total // per_page))
    }

ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'webp', 'gif'}
def allowed_file(filename: str):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXT

# ═══════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/api/auth/signup', methods=['POST'])
def signup():
    data = request.get_json(silent=True) or {}
    first = (data.get('firstName') or '').strip()
    last  = (data.get('lastName')  or '').strip()
    email = (data.get('email')     or '').strip().lower()
    pwd   = data.get('password', '')
    phone = (data.get('phone')     or '').strip()
    utype = data.get('userType', 'gardener')

    if not all([first, last, email, pwd]):
        return jsonify({'error': 'All required fields must be filled'}), 400
    if not EMAIL_RE.match(email):
        return jsonify({'error': 'Invalid email address'}), 400
    if len(pwd) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    db = get_db()
    if db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone():
        return jsonify({'error': 'An account with this email already exists'}), 409

    hashed, salt = hash_password(pwd)
    cur = db.execute(
        'INSERT INTO users (first_name,last_name,email,password,salt,user_type,phone) VALUES (?,?,?,?,?,?,?)',
        (first, last, email, hashed, salt, utype, phone)
    )
    db.commit()
    user_id = cur.lastrowid
    token = create_token(user_id)

    resp = jsonify({
        'message': 'Account created successfully!',
        'token': token,
        'user': {
            'id': user_id, 'firstName': first, 'lastName': last,
            'email': email, 'userType': utype, 'avatar': first[0].upper()
        }
    })
    resp.set_cookie('token', token, httponly=True, samesite='Lax', max_age=72*3600)
    return resp, 201


@app.route('/api/auth/login', methods=['POST'])
def login():
    data  = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    pwd   = data.get('password', '')

    if not email or not pwd:
        return jsonify({'error': 'Email and password are required'}), 400

    db   = get_db()
    user = db.execute('SELECT * FROM users WHERE email=?', (email,)).fetchone()
    if not user:
        return jsonify({'error': 'Invalid email or password'}), 401

    hashed, _ = hash_password(pwd, user['salt'])
    if not hmac.compare_digest(hashed, user['password']):
        return jsonify({'error': 'Invalid email or password'}), 401

    db.execute("UPDATE users SET last_login=datetime('now') WHERE id=?", (user['id'],))
    db.commit()

    token = create_token(user['id'], bool(user['is_admin']))
    resp  = jsonify({
        'message': 'Login successful!',
        'token': token,
        'user': {
            'id': user['id'],
            'firstName': user['first_name'],
            'lastName':  user['last_name'],
            'email':     user['email'],
            'userType':  user['user_type'],
            'isAdmin':   bool(user['is_admin']),
            'avatar':    user['first_name'][0].upper()
        }
    })
    resp.set_cookie('token', token, httponly=True, samesite='Lax', max_age=72*3600)
    return resp


@app.route('/api/auth/logout', methods=['POST'])
def logout():
    resp = jsonify({'message': 'Logged out'})
    resp.delete_cookie('token')
    return resp


@app.route('/api/auth/me', methods=['GET'])
@require_auth
def me():
    db   = get_db()
    user = db.execute('SELECT * FROM users WHERE id=?', (g.user_id,)).fetchone()
    if not user:
        return jsonify({'error': 'User not found'}), 404
    return jsonify({
        'id':        user['id'],
        'firstName': user['first_name'],
        'lastName':  user['last_name'],
        'email':     user['email'],
        'userType':  user['user_type'],
        'phone':     user['phone'],
        'isAdmin':   bool(user['is_admin']),
        'createdAt': user['created_at'],
        'lastLogin': user['last_login'],
        'avatar':    user['first_name'][0].upper()
    })


@app.route('/api/auth/forgot-password', methods=['POST'])
def forgot_password():
    data  = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    db   = get_db()
    user = db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
    # Always return success to prevent email enumeration
    if user:
        token  = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        db.execute(
            'INSERT INTO password_resets (user_id, token, expires_at) VALUES (?,?,?)',
            (user['id'], token, expires)
        )
        db.commit()
        # In production: send email with reset link
        print(f'[RESET TOKEN] user={user["id"]} token={token}')

    return jsonify({'message': 'If that email exists, a reset link has been sent.'})


@app.route('/api/auth/reset-password', methods=['POST'])
def reset_password():
    data    = request.get_json(silent=True) or {}
    token   = data.get('token', '')
    new_pwd = data.get('password', '')
    if not token or len(new_pwd) < 6:
        return jsonify({'error': 'Token and new password (min 6 chars) required'}), 400

    db  = get_db()
    row = db.execute(
        "SELECT * FROM password_resets WHERE token=? AND used=0 AND expires_at > datetime('now')",
        (token,)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Invalid or expired reset token'}), 400

    hashed, salt = hash_password(new_pwd)
    db.execute('UPDATE users SET password=?,salt=? WHERE id=?', (hashed, salt, row['user_id']))
    db.execute('UPDATE password_resets SET used=1 WHERE id=?', (row['id'],))
    db.commit()
    return jsonify({'message': 'Password reset successfully!'})


# ═══════════════════════════════════════════════════════════════
#  PROFILE
# ═══════════════════════════════════════════════════════════════

@app.route('/api/profile', methods=['GET'])
@require_auth
def get_profile():
    db   = get_db()
    user = db.execute('SELECT * FROM users WHERE id=?', (g.user_id,)).fetchone()
    detections_count = db.execute(
        'SELECT COUNT(*) FROM detections WHERE user_id=?', (g.user_id,)
    ).fetchone()[0]
    return jsonify({
        'id':              user['id'],
        'firstName':       user['first_name'],
        'lastName':        user['last_name'],
        'email':           user['email'],
        'userType':        user['user_type'],
        'phone':           user['phone'],
        'createdAt':       user['created_at'],
        'lastLogin':       user['last_login'],
        'avatar':          user['first_name'][0].upper(),
        'detectionsCount': detections_count
    })


@app.route('/api/profile', methods=['PUT'])
@require_auth
def update_profile():
    data = request.get_json(silent=True) or {}
    first = (data.get('firstName') or '').strip()
    last  = (data.get('lastName')  or '').strip()
    phone = (data.get('phone')     or '').strip()
    utype = data.get('userType')

    if not first or not last:
        return jsonify({'error': 'First name and last name are required'}), 400

    db = get_db()
    db.execute(
        'UPDATE users SET first_name=?,last_name=?,phone=?,user_type=? WHERE id=?',
        (first, last, phone, utype, g.user_id)
    )
    db.commit()
    return jsonify({'message': 'Profile updated successfully!'})


@app.route('/api/profile/password', methods=['PUT'])
@require_auth
def change_password():
    data    = request.get_json(silent=True) or {}
    current = data.get('currentPassword', '')
    new_pwd = data.get('newPassword', '')

    if not current or len(new_pwd) < 6:
        return jsonify({'error': 'Current password and new password (min 6) required'}), 400

    db   = get_db()
    user = db.execute('SELECT * FROM users WHERE id=?', (g.user_id,)).fetchone()
    hashed, _ = hash_password(current, user['salt'])
    if not hmac.compare_digest(hashed, user['password']):
        return jsonify({'error': 'Current password is incorrect'}), 400

    new_hashed, new_salt = hash_password(new_pwd)
    db.execute('UPDATE users SET password=?,salt=? WHERE id=?', (new_hashed, new_salt, g.user_id))
    db.commit()
    return jsonify({'message': 'Password changed successfully!'})


# ═══════════════════════════════════════════════════════════════
#  DETECTION
# ═══════════════════════════════════════════════════════════════

@app.route('/api/detect/predict', methods=['POST'])
@require_auth
def predict():
    """
    Run the trained MobileNetV2 model on an uploaded image.
    Expects multipart/form-data with an 'image' field.
    Returns: { disease, confidence (0-100), recommendations[] }
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid or missing image file'}), 400

    try:
        import numpy as np
        model = get_model()
        img_bytes = io.BytesIO(file.read())
        tensor    = preprocess_image(img_bytes)
        preds     = model.predict(tensor, verbose=0)[0]          # shape (3,)
        idx       = int(np.argmax(preds))
        disease   = CLASS_NAMES[idx]
        confidence = round(float(preds[idx]) * 100, 1)
        recommendations = DISEASE_RECOMMENDATIONS.get(disease, [])

        return jsonify({
            'disease':        disease,
            'confidence':     confidence,
            'recommendations': recommendations,
        })
    except RuntimeError as e:
        # Model file missing — return a helpful message instead of 500
        return jsonify({'error': str(e)}), 503
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 500


@app.route('/api/detect', methods=['POST'])
@require_auth
def detect():
    plant_type   = request.form.get('plantType', 'Unknown')
    disease_name = request.form.get('diseaseName', '')
    confidence   = request.form.get('confidence', 0.0)
    recs         = request.form.get('recommendations', '[]')
    notes        = request.form.get('notes', '')

    filename = None
    if 'image' in request.files:
        file = request.files['image']
        if file and file.filename and allowed_file(file.filename):
            ext      = file.filename.rsplit('.', 1)[1].lower()
            filename = f'{g.user_id}_{uuid.uuid4().hex}.{ext}'
            file.save(os.path.join(UPLOAD_DIR, filename))

    db  = get_db()
    cur = db.execute(
        '''INSERT INTO detections
           (user_id, image_filename, plant_type, disease_name, confidence, recommendations, notes)
           VALUES (?,?,?,?,?,?,?)''',
        (g.user_id, filename, plant_type, disease_name, float(confidence), recs, notes)
    )
    db.commit()
    detection_id = cur.lastrowid

    return jsonify({
        'message': 'Detection saved successfully!',
        'id': detection_id,
        'imageUrl': f'/api/uploads/{filename}' if filename else None
    }), 201


@app.route('/api/detections', methods=['GET'])
@require_auth
def get_detections():
    page = request.args.get('page', 1)
    db   = get_db()
    rows = db.execute(
        'SELECT * FROM detections WHERE user_id=? ORDER BY created_at DESC',
        (g.user_id,)
    ).fetchall()

    result = paginate(rows, page)
    # Parse recommendations JSON in each item
    for item in result['items']:
        try:
            item['recommendations'] = json.loads(item['recommendations'] or '[]')
        except Exception:
            item['recommendations'] = []
        if item.get('image_filename'):
            item['imageUrl'] = f'/api/uploads/{item["image_filename"]}'
    return jsonify(result)


@app.route('/api/detections/<int:did>', methods=['GET'])
@require_auth
def get_detection(did):
    db  = get_db()
    row = db.execute(
        'SELECT * FROM detections WHERE id=? AND user_id=?', (did, g.user_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Not found'}), 404
    item = dict(row)
    try:
        item['recommendations'] = json.loads(item['recommendations'] or '[]')
    except Exception:
        item['recommendations'] = []
    if item.get('image_filename'):
        item['imageUrl'] = f'/api/uploads/{item["image_filename"]}'
    return jsonify(item)


@app.route('/api/detections/<int:did>', methods=['DELETE'])
@require_auth
def delete_detection(did):
    db  = get_db()
    row = db.execute(
        'SELECT * FROM detections WHERE id=? AND user_id=?', (did, g.user_id)
    ).fetchone()
    if not row:
        return jsonify({'error': 'Not found'}), 404

    if row['image_filename']:
        try:
            os.remove(os.path.join(UPLOAD_DIR, row['image_filename']))
        except FileNotFoundError:
            pass

    db.execute('DELETE FROM detections WHERE id=?', (did,))
    db.commit()
    return jsonify({'message': 'Detection deleted'})


@app.route('/api/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(UPLOAD_DIR, filename)


# ═══════════════════════════════════════════════════════════════
#  CONTACT
# ═══════════════════════════════════════════════════════════════

@app.route('/api/contact', methods=['POST'])
@require_auth
def submit_contact():
    data    = request.get_json(silent=True) or {}
    name    = (data.get('name')    or '').strip()
    email   = (data.get('email')   or '').strip().lower()
    subject = (data.get('subject') or '').strip()
    message = (data.get('message') or '').strip()

    if not all([name, email, subject, message]):
        return jsonify({'error': 'All fields are required'}), 400

    db = get_db()
    db.execute(
        'INSERT INTO contact_messages (user_id,name,email,subject,message) VALUES (?,?,?,?,?)',
        (g.user_id, name, email, subject, message)
    )
    db.commit()
    return jsonify({'message': 'Message sent successfully! We will get back to you soon.'}), 201


# ═══════════════════════════════════════════════════════════════
#  NEWSLETTER
# ═══════════════════════════════════════════════════════════════

@app.route('/api/newsletter/subscribe', methods=['POST'])
def newsletter_subscribe():
    data  = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    if not EMAIL_RE.match(email):
        return jsonify({'error': 'Invalid email address'}), 400

    # Detect if logged in (optional)
    user_id = None
    auth    = request.headers.get('Authorization', '')
    token   = auth.removeprefix('Bearer ').strip() or request.cookies.get('token', '')
    payload = verify_token(token)
    if payload:
        user_id = payload['sub']

    db = get_db()
    existing = db.execute('SELECT id,subscribed FROM newsletter WHERE email=?', (email,)).fetchone()
    if existing:
        if existing['subscribed']:
            return jsonify({'message': "You're already subscribed! 🌿"})
        db.execute('UPDATE newsletter SET subscribed=1 WHERE email=?', (email,))
        db.commit()
        return jsonify({'message': 'Welcome back! Re-subscribed successfully 🌿'})

    db.execute('INSERT INTO newsletter (email, user_id) VALUES (?,?)', (email, user_id))
    db.commit()
    return jsonify({'message': "You're in! Welcome to the Verdana community 🌿"}), 201


@app.route('/api/newsletter/unsubscribe', methods=['POST'])
def newsletter_unsubscribe():
    data  = request.get_json(silent=True) or {}
    email = (data.get('email') or '').strip().lower()
    db    = get_db()
    db.execute('UPDATE newsletter SET subscribed=0 WHERE email=?', (email,))
    db.commit()
    return jsonify({'message': 'Unsubscribed successfully.'})


# ═══════════════════════════════════════════════════════════════
#  ADMIN
# ═══════════════════════════════════════════════════════════════

@app.route('/api/admin/users', methods=['GET'])
@require_auth
@require_admin
def admin_users():
    page = request.args.get('page', 1)
    db   = get_db()
    rows = db.execute(
        'SELECT id,first_name,last_name,email,user_type,is_admin,created_at,last_login FROM users ORDER BY created_at DESC'
    ).fetchall()
    return jsonify(paginate(rows, page))


@app.route('/api/admin/stats', methods=['GET'])
@require_auth
@require_admin
def admin_stats():
    db = get_db()
    stats = {
        'totalUsers':      db.execute('SELECT COUNT(*) FROM users').fetchone()[0],
        'totalDetections': db.execute('SELECT COUNT(*) FROM detections').fetchone()[0],
        'totalMessages':   db.execute('SELECT COUNT(*) FROM contact_messages').fetchone()[0],
        'unreadMessages':  db.execute("SELECT COUNT(*) FROM contact_messages WHERE status='unread'").fetchone()[0],
        'newsletterSubs':  db.execute('SELECT COUNT(*) FROM newsletter WHERE subscribed=1').fetchone()[0],
        'recentUsers':     db.execute(
            "SELECT COUNT(*) FROM users WHERE created_at >= datetime('now','-7 days')"
        ).fetchone()[0],
        'recentDetections': db.execute(
            "SELECT COUNT(*) FROM detections WHERE created_at >= datetime('now','-7 days')"
        ).fetchone()[0],
    }
    return jsonify(stats)


@app.route('/api/admin/messages', methods=['GET'])
@require_auth
@require_admin
def admin_messages():
    page = request.args.get('page', 1)
    db   = get_db()
    rows = db.execute('SELECT * FROM contact_messages ORDER BY created_at DESC').fetchall()
    return jsonify(paginate(rows, page))


@app.route('/api/admin/messages/<int:mid>/read', methods=['PUT'])
@require_auth
@require_admin
def mark_message_read(mid):
    db = get_db()
    db.execute("UPDATE contact_messages SET status='read' WHERE id=?", (mid,))
    db.commit()
    return jsonify({'message': 'Marked as read'})


@app.route('/api/admin/detections', methods=['GET'])
@require_auth
@require_admin
def admin_detections():
    page = request.args.get('page', 1)
    db   = get_db()
    rows = db.execute('''
        SELECT d.*, u.email AS user_email, u.first_name, u.last_name
        FROM detections d
        LEFT JOIN users u ON u.id = d.user_id
        ORDER BY d.created_at DESC
    ''').fetchall()
    return jsonify(paginate(rows, page))


@app.route('/api/admin/create-admin', methods=['POST'])
@require_auth
@require_admin
def create_admin():
    data  = request.get_json(silent=True) or {}
    first = (data.get('firstName') or '').strip()
    last  = (data.get('lastName')  or '').strip()
    email = (data.get('email')     or '').strip().lower()
    pwd   = data.get('password', '')

    if not all([first, last, email, pwd]):
        return jsonify({'error': 'All fields required'}), 400
    if not EMAIL_RE.match(email):
        return jsonify({'error': 'Invalid email'}), 400
    if len(pwd) < 8:
        return jsonify({'error': 'Password must be at least 8 characters'}), 400

    db = get_db()
    if db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone():
        return jsonify({'error': 'Email already in use'}), 409

    hashed, salt = hash_password(pwd)
    cur = db.execute(
        'INSERT INTO users (first_name,last_name,email,password,salt,user_type,is_admin) VALUES (?,?,?,?,?,?,1)',
        (first, last, email, hashed, salt, 'admin')
    )
    db.commit()
    return jsonify({'message': f'Admin account created for {email}', 'id': cur.lastrowid}), 201


@app.route('/api/admin/newsletter', methods=['GET'])
@require_auth
@require_admin
def admin_newsletter():
    page = request.args.get('page', 1)
    db   = get_db()
    rows = db.execute('SELECT * FROM newsletter ORDER BY created_at DESC').fetchall()
    return jsonify(paginate(rows, page))


# ═══════════════════════════════════════════════════════════════
#  STATIC FILE SERVING  (so you can open purchase.html & prototype.html directly)
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('prototype.html')

@app.route('/purchase.html')
def purchase():
    return render_template('purchase.html')

@app.route('/prototype.html')
def prototype():
    return render_template('prototype.html')


@app.route('/admin')
@app.route('/admin.html')
def admin_page():
    return render_template('admin.html')


@app.route('/profile')
@app.route('/profile.html')
def profile_page():
    return render_template('profile.html')


# ═══════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═══════════════════════════════════════════════════════════════

@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.now(timezone.utc).isoformat()})


# ═══════════════════════════════════════════════════════════════
#  CLI HELPERS
# ═══════════════════════════════════════════════════════════════

def seed_admin(email: str, password: str, first: str = 'Admin', last: str = 'User'):
    """Create an admin user from the command line."""
    init_db()
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    existing = db.execute('SELECT id FROM users WHERE email=?', (email,)).fetchone()
    if existing:
        print(f'⚠  User {email} already exists (id={existing["id"]}). Updating to admin…')
        db.execute('UPDATE users SET is_admin=1 WHERE email=?', (email,))
        db.commit()
        print(f'✓  {email} is now an admin.')
        db.close()
        return
    hashed, salt = hash_password(password)
    cur = db.execute(
        'INSERT INTO users (first_name,last_name,email,password,salt,user_type,is_admin) VALUES (?,?,?,?,?,?,1)',
        (first, last, email, hashed, salt, 'admin')
    )
    db.commit()
    print(f'✓  Admin created: {email} (id={cur.lastrowid})')
    db.close()


# ─── Boot ─────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    # ── CLI: python app.py create-admin email password [first] [last]
    if len(sys.argv) >= 2 and sys.argv[1] == 'create-admin':
        if len(sys.argv) < 4:
            print('Usage: python app.py create-admin <email> <password> [first] [last]')
            sys.exit(1)
        seed_admin(
            email    = sys.argv[2],
            password = sys.argv[3],
            first    = sys.argv[4] if len(sys.argv) > 4 else 'Admin',
            last     = sys.argv[5] if len(sys.argv) > 5 else 'User'
        )
        sys.exit(0)

    init_db()
    print('\n🌿 Verdana Backend  ·  http://localhost:5000\n')
    print('  Pages:')
    print('  /                   → Home (prototype.html)')
    print('  /purchase.html      → Shop')
    print('  /profile.html       → User Profile')
    print('  /admin.html         → Admin Dashboard\n')
    print('  Auth API:')
    print('  POST /api/auth/signup · login · logout · me')
    print('  POST /api/auth/forgot-password · reset-password\n')
    print('  User API:')
    print('  GET|PUT  /api/profile')
    print('  PUT      /api/profile/password\n')
    print('  Detection API:')
    print('  POST /api/detect')
    print('  GET|DELETE /api/detections · /api/detections/<id>\n')
    print('  Other:')
    print('  POST /api/contact')
    print('  POST /api/newsletter/subscribe · unsubscribe\n')
    print('  Admin API (requires admin token):')
    print('  GET  /api/admin/stats · users · detections · messages · newsletter')
    print('  PUT  /api/admin/messages/<id>/read')
    print('  POST /api/admin/create-admin\n')
    print('  CLI:')
    print('  python app.py create-admin admin@verdana.com mypassword\n')
    app.run(debug=True, host='0.0.0.0', port=5000)