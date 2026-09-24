import os
import re
import json
import socket
import sqlite3
import hashlib
import ipaddress
import requests
from datetime import datetime
from openai import OpenAI
import plotly
import plotly.graph_objects as go
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from authlib.integrations.flask_client import OAuth
from pdfminer.high_level import extract_text
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from urllib.parse import quote_plus, urlparse
from bs4 import BeautifulSoup

load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-fallback-change-me")

UPLOAD_FOLDER = 'uploads'
DB_PATH = 'cvalign_data.db'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

ADMIN_EMAILS = {e.strip().lower() for e in os.getenv("ADMIN_EMAILS", "").split(",") if e.strip()}
FREE_SIGNUP_CREDITS = int(os.getenv("FREE_SIGNUP_CREDITS", "5"))
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024   # 5 MB upload cap

# Credit cost of every action that calls OpenAI (override in .env)
COSTS = {
    "cv_analysis": int(os.getenv("COST_CV_ANALYSIS", "1")),
    "ats_scan":    int(os.getenv("COST_ATS_SCAN", "1")),
    "job_cv":      int(os.getenv("COST_JOB_CV_UPLOAD", "1")),   # reading a CV for job matching
    "job_match":   int(os.getenv("COST_JOB_MATCH", "1")),       # scoring one page of results
    "job_detail":  int(os.getenv("COST_JOB_DETAIL", "1")),      # opening one job (not cached)
}

# --- LOGIN MANAGER ---
login_manager = LoginManager()
login_manager.init_app(app)


@login_manager.unauthorized_handler
def unauthorized():
    if request.method != 'GET':          # fetch()/API calls get JSON instead of a redirect
        return jsonify({"error": "Please sign in to continue.", "auth_required": True}), 401
    session['next_url'] = request.path   # come back here after sign-in
    return redirect(url_for('dashboard_page', signin='required'))

# --- GOOGLE OAUTH ---
oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=os.getenv("GOOGLE_CLIENT_ID"),
    client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

SUPPORTED_COUNTRIES = {
    'India': 'in', 'Australia': 'au', 'UAE': 'ae', 'Austria': 'at',
    'Belgium': 'be', 'Brazil': 'br', 'Canada': 'ca', 'Switzerland': 'ch',
    'Germany': 'de', 'Spain': 'es', 'France': 'fr', 'UK': 'gb',
    'Italy': 'it', 'Mexico': 'mx', 'Netherlands': 'nl', 'New Zealand': 'nz',
    'Poland': 'pl', 'Singapore': 'sg', 'USA': 'us', 'South Africa': 'za'
}

ALLOWED_EXTENSIONS = {'.pdf'}


def allowed_file(filename):
    return os.path.splitext(filename)[1].lower() in ALLOWED_EXTENSIONS


class User(UserMixin):
    def __init__(self, id, email, name, credits_remaining, is_admin):
        self.id = id
        self.email = email
        self.name = name
        self.credits_remaining = credits_remaining
        self.is_admin = is_admin


@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, name, credits_remaining FROM users WHERE id = ?", (user_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    is_admin = row[1].lower() in ADMIN_EMAILS
    return User(row[0], row[1], row[2], row[3], is_admin)


# --- DATABASE LOGIC ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS global_benchmarks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT NOT NULL,
            country TEXT NOT NULL,
            required_skills TEXT NOT NULL,
            UNIQUE(role_name, country)
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS processed_cvs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_name TEXT,
            target_role TEXT,
            country TEXT,
            match_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            google_id TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            name TEXT,
            credits_remaining INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    # NEW: CV profiles for job matching
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS cv_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            profile_json TEXT NOT NULL,
            cv_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    try:   # upgrade an existing table created before user_id existed
        cursor.execute("ALTER TABLE cv_profiles ADD COLUMN user_id INTEGER")
    except sqlite3.OperationalError:
        pass
    # NEW: cache of extracted job details
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_details_cache (
            url_hash TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()


def get_or_create_user(google_id, email, name):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id, email, name, credits_remaining FROM users WHERE google_id = ?", (google_id,))
    row = cursor.fetchone()

    if row:
        conn.close()
        is_admin = row[1].lower() in ADMIN_EMAILS
        return User(row[0], row[1], row[2], row[3], is_admin)

    starting_credits = 999999 if email.lower() in ADMIN_EMAILS else FREE_SIGNUP_CREDITS
    cursor.execute('''
        INSERT INTO users (google_id, email, name, credits_remaining)
        VALUES (?, ?, ?, ?)
    ''', (google_id, email, name, starting_credits))
    conn.commit()
    new_id = cursor.lastrowid
    conn.close()
    is_admin = email.lower() in ADMIN_EMAILS
    return User(new_id, email, name, starting_credits, is_admin)


def charge_credits(user, amount):
    """Atomically deduct `amount`. Admins are never charged. False if not enough credits."""
    if user.is_admin or amount <= 0:
        return True
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "UPDATE users SET credits_remaining = credits_remaining - ? WHERE id = ? AND credits_remaining >= ?",
        (amount, user.id, amount))
    conn.commit()
    ok = cur.rowcount == 1
    conn.close()
    return ok


def refund_credits(user, amount):
    if user.is_admin or amount <= 0:
        return
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE users SET credits_remaining = credits_remaining + ? WHERE id = ?", (amount, user.id))
    conn.commit()
    conn.close()


def credits_left(user):
    if user.is_admin:
        return "Unlimited"
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT credits_remaining FROM users WHERE id = ?", (user.id,)).fetchone()
    conn.close()
    return row[0] if row else 0


OUT_OF_CREDITS = {"error": "You're out of credits. Please top up to continue.", "out_of_credits": True}


def generate_market_skills_ai(target_role, country):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a labor market analyst. Return ONLY JSON."},
                {"role": "user", "content": f"""
                List the 6-8 most important, specific skills required for a
                "{target_role}" role in {country} in 2026. Be specific to THIS
                role — do not default to generic data/tech skills unless the
                role is actually a data/tech role.

                Return JSON: {{"skills": ["skill1", "skill2", ...]}}
                """}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        skills = result.get('skills', [])
        return skills if skills else ["Communication", "Problem Solving", "Time Management"]
    except Exception as e:
        print(f"Benchmark Generation Error: {e}")
        return ["Communication", "Problem Solving", "Time Management"]


def fetch_dynamic_benchmark(target_role, country):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT required_skills FROM global_benchmarks WHERE role_name LIKE ? AND country = ?",
                   (f'%{target_role}%', country))
    row = cursor.fetchone()

    if row:
        conn.close()
        return [skill.strip() for skill in row[0].split(',')]

    generated_skills = generate_market_skills_ai(target_role, country)
    try:
        cursor.execute('''
            INSERT INTO global_benchmarks (role_name, country, required_skills)
            VALUES (?, ?, ?)
        ''', (target_role, country, ", ".join(generated_skills)))
        conn.commit()
    except Exception as e:
        print(f"Benchmark Cache Save Error: {e}")
    finally:
        conn.close()

    return generated_skills


def save_analysis_to_db(name, role, country, score):
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO processed_cvs (candidate_name, target_role, country, match_score)
            VALUES (?, ?, ?, ?)
        ''', (name, role, country, score))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"Database Save Error: {e}")


def get_ai_cv_intelligence(cv_text, target_role, country, market_skills):
    try:
        skills_str = ", ".join(market_skills)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[
                {"role": "system", "content": "You are a recruitment expert. Return ONLY JSON."},
                {"role": "user", "content": f"""
                Analyze this CV for a {target_role} role in {country}.

                The market requires these specific skills: {skills_str}

                Return JSON with exactly these keys:
                - candidate_name: string
                - current_role: string
                - years_of_experience: number
                - identified_skills: array of skills the candidate ACTUALLY has, found in the CV text
                - matched_skills: array — subset of [{skills_str}] that the candidate demonstrably has
                - gap_analysis: array — subset of [{skills_str}] that the candidate is missing, based on matched_skills above
                - match_score: number 0-100, calculated as (len(matched_skills) / len({market_skills})) * 100, adjusted slightly for years of experience

                CV: {cv_text[:4000]}
                """}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result.setdefault('candidate_name', 'Candidate')
        result.setdefault('current_role', 'Not specified')
        result.setdefault('years_of_experience', 0)
        result.setdefault('identified_skills', [])
        result.setdefault('matched_skills', [])
        result.setdefault('gap_analysis', [s for s in market_skills if s not in result.get('matched_skills', [])])
        result.setdefault('match_score', 0)
        return result
    except Exception as e:
        print(f"OpenAI Error: {e}")
        raise e


def get_ats_intelligence(cv_text, jd_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are an ATS optimization expert. Return ONLY JSON."},
                {"role": "user", "content": f"""
                Perform a deep-scan comparison of this CV against the provided Job Description (JD).
                Return JSON with: name, score (0-100), keywords (20-25 missing terms),
                rewrites (6-10 objects with 'original' and 'suggested' keys), weak_verbs (array).

                CV: {cv_text[:4000]}
                JD: {jd_text[:3000]}
                """}
            ],
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        result.setdefault('name', 'Candidate')
        result.setdefault('score', 0)
        result.setdefault('keywords', [])
        result.setdefault('rewrites', [])
        result.setdefault('weak_verbs', [])
        return result
    except Exception as e:
        print(f"Deep Scan AI Error: {e}")
        raise e


def generate_radar_chart(matched_skills, market_skills):
    matched_lower = {s.lower() for s in matched_skills} if matched_skills else set()
    user_values = [1 if skill.lower() in matched_lower else 0.2 for skill in market_skills]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=user_values, theta=market_skills, fill='toself', name='Your Profile', line_color='#3b82f6'))
    fig.add_trace(go.Scatterpolar(r=[1] * len(market_skills), theta=market_skills, name='Market Demand', line_color='#10b981'))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1]), angularaxis=dict(color="white", tickfont=dict(size=10))),
        showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color="white"),
        margin=dict(l=60, r=60, t=20, b=20)
    )
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


# =====================================================================
# JOB ENGINE HELPERS (NEW)
# =====================================================================
def clean_html(text):
    """Strip HTML tags and collapse whitespace."""
    return re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', text or '')).strip()


def is_safe_url(url):
    """Block SSRF: only public http(s) hosts."""
    try:
        p = urlparse(url)
        if p.scheme not in ('http', 'https') or not p.hostname:
            return False
        for info in socket.getaddrinfo(p.hostname, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                return False
        return True
    except Exception:
        return False


AGGREGATOR_HOSTS = ('jooble.', 'adzuna.')


def fetch_page_text(url):
    """Returns (text, final_url). text is '' when the page can't be read."""
    if not is_safe_url(url):
        return '', ''
    try:
        r = requests.get(url, timeout=8, allow_redirects=True, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            "Accept-Language": "en"})
        final_url = r.url if is_safe_url(r.url) else ''      # re-check after redirects
        if not final_url:
            return '', ''
        if r.status_code != 200 or 'html' not in r.headers.get('Content-Type', ''):
            return '', final_url
        soup = BeautifulSoup(r.text[:1_500_000], 'html.parser')
        for t in soup(['script', 'style', 'nav', 'footer', 'header', 'noscript', 'svg', 'form']):
            t.decompose()
        text = re.sub(r'\s+', ' ', soup.get_text(' ')).strip()
        return (text[:6000] if len(text) > 300 else ''), final_url
    except Exception as e:
        print(f"Page fetch error: {e}")
        return '', ''


def quick_match(text, skills):
    """Cheap keyword-overlap score for the list view (no AI cost)."""
    if not skills:
        return None
    t = (text or '').lower()
    hits = [s for s in skills if s.lower() in t]
    return min(95, len(hits) * 20)


def get_cv_profile():
    cv_id = session.get('cv_id')
    if not cv_id or not current_user.is_authenticated:
        return None
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT profile_json, cv_text FROM cv_profiles WHERE id = ? AND user_id = ?",
                       (cv_id, current_user.id)).fetchone()
    conn.close()
    if not row:
        return None
    profile = json.loads(row[0])
    profile['_text'] = row[1]
    return profile


def extract_job_intel(title, company, snippet, page_text, profile):
    source_text = page_text or snippet
    cv_block = ""
    match_spec = ""
    if profile:
        cv_block = (f"\nCANDIDATE PROFILE: skills={profile.get('skills')}, "
                    f"years_experience={profile.get('years_of_experience')}, "
                    f"titles={profile.get('titles')}\nCV EXCERPT: {profile['_text'][:2500]}")
        match_spec = ('- match: object with score (0-100 integer), verdict (one short sentence), '
                      'matched_skills (array), missing_skills (array), tip (one actionable sentence)')
    response = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You extract structured data from job postings. "
             "ALWAYS write every output value in English, translating from the posting's language "
             "(e.g. German, French, Spanish) when needed. Keep company names and tool or product names "
             "(Power BI, SQL, Python) unchanged. "
             "Use ONLY the text given; use null / [] when something is not stated. Never invent. Return ONLY JSON."},
            {"role": "user", "content": f"""
Job: {title} at {company}
JOB TEXT: {source_text}
{cv_block}

Return JSON with:
- title_en: the job title translated into English
- summary: 2-3 sentence plain-English summary
- experience_required: string (e.g. "5+ years") or null
- education: string or null
- employment_type: string or null
- seniority: string or null
- salary: string or null
- key_skills: array (max 10)
- responsibilities: array (max 6, short)
- nice_to_have: array (max 5)
- benefits: array (max 5)
- company_website: the employer's own website or careers page as a full URL starting with http,
  ONLY if it is literally written in the JOB TEXT, otherwise null
{match_spec}
All string values in your JSON must be in English.
"""}])
    return json.loads(response.choices[0].message.content)


# --- AUTH ROUTES ---
@app.route('/dashboard')
def dashboard_page():
    return render_template('dashboard.html')


@app.route('/login')
def login():
    redirect_uri = url_for('auth_callback', _external=True)
    return google.authorize_redirect(redirect_uri)


@app.route('/auth/callback')
def auth_callback():
    try:
        token = google.authorize_access_token()
        user_info = token.get('userinfo')
        if not user_info:
            return redirect(url_for('dashboard_page'))
        user = get_or_create_user(
            google_id=user_info['sub'],
            email=user_info['email'],
            name=user_info.get('name', user_info['email'])
        )
        login_user(user)
    except Exception as e:
        print(f"OAuth callback error: {e}")
        return redirect(url_for('dashboard_page'))
    nxt = session.pop('next_url', None)
    return redirect('/jobs' if nxt == '/jobs' else url_for('dashboard_page'))


@app.route('/logout')
@login_required
def logout():
    session.pop('cv_id', None)
    logout_user()
    return redirect(url_for('dashboard_page'))


@app.route('/api/pricing')
def api_pricing():
    return jsonify({"costs": COSTS, "free_credits": FREE_SIGNUP_CREDITS})


@app.route('/api/me')
def api_me():
    if current_user.is_authenticated:
        return jsonify({
            "logged_in": True,
            "email": current_user.email,
            "name": current_user.name,
            "credits_remaining": "Unlimited" if current_user.is_admin else current_user.credits_remaining
        })
    return jsonify({"logged_in": False})


# --- PAGE ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/ats')
def ats_page():
    return render_template('ats.html')


@app.route('/jobs')
def job_engine_page():
    return render_template('job_engine.html')


@app.route('/fetch-jobs', methods=['POST'])
@login_required
def fetch_jobs():
    role = request.form.get('role', 'Senior Data Analyst')
    location = request.form.get('location', '')
    country_name = request.form.get('country', 'India')
    page = request.form.get('page', 1)
    CURRENCY = {'gb': '£', 'us': '$', 'in': '₹', 'au': 'A$', 'ca': 'C$', 'ae': 'AED ', 'de': '€', 'fr': '€',
            'es': '€', 'it': '€', 'nl': '€', 'at': '€', 'be': '€', 'ch': 'CHF ', 'pl': 'PLN ',
            'sg': 'S$', 'nz': 'NZ$', 'za': 'R', 'br': 'R$', 'mx': 'MX$'}

    # NEW: optional CV skills for quick-match badges
    profile = get_cv_profile()
    cv_skills = profile.get('skills', []) if profile else []

    job_results = []

    if country_name == 'UAE':
        jooble_key = os.getenv("JOOBLE_API_KEY")
        if not jooble_key:
            return jsonify({"error": "Jooble API Key missing"}), 500
        url = f"https://jooble.org/api/{jooble_key.strip()}"
        payload = {"keywords": role, "location": location if location else "UAE", "page": str(page)}
        try:
            response = requests.post(url, json=payload, timeout=10)
            if response.status_code != 200:
                return jsonify({"error": f"Jooble Error: {response.status_code}"}), 500
            data = response.json()
            for job in data.get('jobs', []):
                company_name = job.get('company', 'Unknown')
                safe_company = company_name.replace('"', '')
                xray_query = f'site:linkedin.com/in/ "{safe_company}" (Recruiter OR "Hiring Manager" OR "Talent Acquisition" OR "HRBP")'
                snippet = clean_html(job.get('snippet', ''))
                job_results.append({
                    "title": job.get('title'), "company": job.get('company'),
                    "location": job.get('location'), "link": job.get('link'),
                    "updated": job.get('updated'), "salary": job.get('salary') or 'Market Rate',
                    "snippet": snippet,
                    "quick_match": quick_match(f"{job.get('title', '')} {snippet}", cv_skills),
                    "xray_link": f"https://www.google.com/search?q={quote_plus(xray_query)}"
                })
        except Exception as e:
            return jsonify({"error": f"Jooble failure: {str(e)}"}), 500
    else:
        country_code = SUPPORTED_COUNTRIES.get(country_name, 'in')
        app_id = os.getenv("ADZUNA_APP_ID")
        app_key = os.getenv("ADZUNA_APP_KEY")
        url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/{page}"
        params = {"app_id": app_id, "app_key": app_key, "results_per_page": 15, "what": role, "content-type": "application/json"}
        if location:
            params["where"] = location
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            for job in data.get('results', []):
                company_name = job.get('company', {}).get('display_name', 'Unknown')
                safe_company = company_name.replace('"', '')
                xray_query = f'site:linkedin.com/in/ "{safe_company}" (Recruiter OR "Hiring Manager")'
                snippet = clean_html(job.get('description', ''))
                salary_min = job.get('salary_min')
                job_results.append({
                    "title": job.get('title'), "company": company_name,
                    "location": job.get('location', {}).get('display_name'),
                    "link": job.get('redirect_url'), "updated": job.get('created'),
                    "salary": f"From {CURRENCY.get(country_code, '')}{int(salary_min):,}" if salary_min else "Not disclosed",
                    "snippet": snippet,
                    #"quick_match": quick_match(f"{job.get('title', '')} {snippet}", cv_skills),
                    "xray_link": f"https://www.google.com/search?q={quote_plus(xray_query)}"
                })
        except Exception as e:
            return jsonify({"error": f"Adzuna failure: {str(e)}"}), 500

    return jsonify({"jobs": job_results})


# =====================================================================
# JOB ENGINE ROUTES (NEW)
# =====================================================================
@app.route('/upload-cv-profile', methods=['POST'])
@login_required
def upload_cv_profile():
    file = request.files.get('file')
    if not file or not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Please upload a PDF"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, f"{os.urandom(6).hex()}_{secure_filename(file.filename)}")
    file.save(filepath)
    charged = False
    try:
        text = extract_text(filepath)
        if not text.strip():
            return jsonify({"error": "Could not read text from this PDF (scanned image?)"}), 400
        if not charge_credits(current_user, COSTS['job_cv']):
            return jsonify(OUT_OF_CREDITS), 402
        charged = True
        resp = client.chat.completions.create(
            model="gpt-4o-mini", temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "Extract a candidate profile. Write all values in English. Return ONLY JSON."},
                {"role": "user", "content": f"""From this CV return JSON: skills (array, max 25, only skills
actually present), titles (array of job titles held), years_of_experience (number),
suggested_role (string, best-fit target role).\n\nCV: {text[:4000]}"""}])
        profile = json.loads(resp.choices[0].message.content)
        profile.setdefault('skills', [])
        conn = sqlite3.connect(DB_PATH)
        cur = conn.execute("INSERT INTO cv_profiles (user_id, profile_json, cv_text) VALUES (?, ?, ?)",
                           (current_user.id, json.dumps(profile), text[:8000]))
        conn.commit()
        session['cv_id'] = cur.lastrowid
        conn.close()
        return jsonify({"ok": True, "skills": profile['skills'][:12],
                        "suggested_role": profile.get('suggested_role', ''),
                        "years": profile.get('years_of_experience', 0),
                        "credits_remaining": credits_left(current_user)})
    except Exception as e:
        print(f"CV profile error: {e}")
        if charged:
            refund_credits(current_user, COSTS['job_cv'])
        return jsonify({"error": "Could not process CV. Please try again."}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/clear-cv', methods=['POST'])
def clear_cv():
    cv_id = session.pop('cv_id', None)
    if cv_id and current_user.is_authenticated:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("DELETE FROM cv_profiles WHERE id = ? AND user_id = ?", (cv_id, current_user.id))
        conn.commit()
        conn.close()
    return jsonify({"ok": True})


@app.route('/job-detail', methods=['POST'])
@login_required
def job_detail():
    body = request.get_json(silent=True) or {}
    url = str(body.get('link', ''))[:2000]
    title = str(body.get('title', ''))[:200]
    company = str(body.get('company', ''))[:200]
    snippet = clean_html(str(body.get('snippet', '')))[:3000]
    profile = get_cv_profile()

    key = hashlib.sha256(("v3|" + url + (f"|cv{session.get('cv_id')}" if profile else '')).encode()).hexdigest()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT data FROM job_details_cache WHERE url_hash = ?", (key,)).fetchone()
    conn.close()
    if row:                                   # cached = no OpenAI cost = free
        return jsonify(apply_list_score(json.loads(row[0]), url))

    if not charge_credits(current_user, COSTS['job_detail']):
        return jsonify(OUT_OF_CREDITS), 402

    page_text, final_url = fetch_page_text(url) if url else ('', '')
    if not page_text and len(snippet) < 40:
        refund_credits(current_user, COSTS['job_detail'])
        return jsonify({"error": "This listing could not be read. Use the Apply button to open it."}), 200
    try:
        intel = extract_job_intel(title, company, snippet, page_text, profile)
    except Exception as e:
        print(f"Job intel error: {e}")
        refund_credits(current_user, COSTS['job_detail'])
        return jsonify({"error": "Could not analyze this job right now."}), 500

    intel['source'] = 'full_page' if page_text else 'snippet_only'

    host = (urlparse(final_url).hostname or '').lower() if final_url else ''
    is_direct = bool(host) and not any(a in host for a in AGGREGATOR_HOSTS)
    intel['direct_url'] = final_url if is_direct else ''
    intel['direct_host'] = host if is_direct else ''

    site = intel.get('company_website')
    try:
        site_host = (urlparse(site).hostname or '').lower() if site else ''
    except Exception:
        site_host = ''
    src = (page_text or snippet).lower()
    intel['company_website'] = site if (site_host and site_host in src
                                        and not any(a in site_host for a in AGGREGATOR_HOSTS)
                                        and is_safe_url(site)) else None

    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT OR REPLACE INTO job_details_cache (url_hash, data) VALUES (?, ?)",
                 (key, json.dumps(intel)))
    conn.commit()
    conn.close()
    out = apply_list_score(intel, url)
    out['credits_remaining'] = credits_left(current_user)
    return jsonify(out)

# ---------- AI match scoring shared by list + detail panel ----------
def score_cache_key(link, cv_id):
    return hashlib.sha256(f"score|{link}|cv{cv_id}".encode()).hexdigest()


def apply_list_score(intel, link):
    """Make the detail panel show the same % as the list."""
    cv_id = session.get('cv_id')
    if not cv_id or not isinstance(intel.get('match'), dict):
        return intel
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT data FROM job_details_cache WHERE url_hash = ?",
                       (score_cache_key(link, cv_id),)).fetchone()
    conn.close()
    if row:
        intel['match']['score'] = json.loads(row[0])['score']
    return intel


@app.route('/cv-status')
def cv_status():
    profile = get_cv_profile()
    return jsonify({"loaded": bool(profile), "skills": len(profile.get('skills', [])) if profile else 0})

@app.route('/match-jobs', methods=['POST'])
@login_required
def match_jobs():
    profile = get_cv_profile()
    if not profile:
        return jsonify({"scores": {}})
    jobs = (request.get_json(silent=True) or {}).get('jobs', [])[:20]
    cv_id = session.get('cv_id')
    scores, todo = {}, []

    conn = sqlite3.connect(DB_PATH)
    for j in jobs:
        key = score_cache_key(str(j.get('link', '')), cv_id)
        row = conn.execute("SELECT data FROM job_details_cache WHERE url_hash = ?", (key,)).fetchone()
        if row:
            scores[str(j.get('id'))] = json.loads(row[0])['score']
        else:
            todo.append((j, key))
    conn.close()

    if todo:
        if not charge_credits(current_user, COSTS['job_match']):
            return jsonify({**OUT_OF_CREDITS, "scores": scores}), 402
        saved = 0
        try:
            listing = "\n".join(
                f"[{j.get('id')}] {str(j.get('title', ''))[:200]} at {str(j.get('company', ''))[:200]}: "
                f"{clean_html(str(j.get('snippet', '')))[:500]}" for j, _ in todo)
            resp = client.chat.completions.create(
                model="gpt-4o-mini", temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You score how well a candidate fits each job. Return ONLY JSON."},
                    {"role": "user", "content": f"""
CANDIDATE: skills={profile.get('skills')}, years_experience={profile.get('years_of_experience')},
titles={profile.get('titles')}
CV EXCERPT: {profile['_text'][:1500]}

JOBS:
{listing}

Score each job 0-100 using: skill overlap, role/seniority fit, years of experience, industry relevance.
Be consistent and realistic; do not give everyone the same score.
Return JSON: {{"scores": [{{"id": "<id>", "score": <int>}}]}}"""}])
            result = json.loads(resp.choices[0].message.content).get('scores', [])
            by_id = {str(x.get('id')): int(x.get('score', 0)) for x in result}
            conn = sqlite3.connect(DB_PATH)
            for j, key in todo:
                jid = str(j.get('id'))
                s = by_id.get(jid)
                if s is not None:
                    s = max(0, min(100, s))
                    scores[jid] = s
                    conn.execute("INSERT OR REPLACE INTO job_details_cache (url_hash, data) VALUES (?, ?)",
                                 (key, json.dumps({"score": s})))
                    saved += 1
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Match scoring error: {e}")
        if saved == 0:                       # nothing delivered -> no charge
            refund_credits(current_user, COSTS['job_match'])
    return jsonify({"scores": scores, "credits_remaining": credits_left(current_user)})

# --- CV ANALYSIS ROUTES ---
@app.route('/upload-ats', methods=['POST'])
@login_required
def upload_ats():
    jd_text = request.form.get('jd_text', '')
    if 'file' not in request.files or not jd_text.strip():
        return jsonify({"error": "Missing CV file or JD text"}), 400
    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, f"{os.urandom(6).hex()}_{secure_filename(file.filename)}")
    file.save(filepath)
    charged = False
    try:
        raw_text = extract_text(filepath)
        if not raw_text.strip():
            return jsonify({"error": "Could not read text from this PDF. Is it a scanned image?"}), 400
        if not charge_credits(current_user, COSTS['ats_scan']):
            return jsonify(OUT_OF_CREDITS), 402
        charged = True
        intel = get_ats_intelligence(raw_text, jd_text)
        return jsonify({
            "score": intel['score'],
            "status": "High Match" if intel['score'] > 80 else "Optimization Needed",
            "name": intel['name'], "keywords": intel['keywords'],
            "rewrites": intel['rewrites'], "weak_verbs": intel.get('weak_verbs', []),
            "credits_remaining": credits_left(current_user)
        })
    except Exception as e:
        print(f"ATS Error: {str(e)}")
        if charged:
            refund_credits(current_user, COSTS['ats_scan'])
        return jsonify({"error": "Analysis failed. Please try again."}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    target_country = request.form.get('country', 'UAE')
    target_role = request.form.get('role', 'Data Analyst')

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    file = request.files['file']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, f"{os.urandom(6).hex()}_{secure_filename(file.filename)}")
    file.save(filepath)
    charged = False
    try:
        raw_text = extract_text(filepath)
        if not raw_text.strip():
            return jsonify({"error": "Could not read text from this PDF. Is it a scanned image?"}), 400
        if not charge_credits(current_user, COSTS['cv_analysis']):
            return jsonify(OUT_OF_CREDITS), 402
        charged = True

        market_skills = fetch_dynamic_benchmark(target_role, target_country)
        intel = get_ai_cv_intelligence(raw_text, target_role, target_country, market_skills)
        graph_json = generate_radar_chart(intel['matched_skills'], market_skills)
        save_analysis_to_db(intel.get('candidate_name', 'Unknown'), target_role, target_country, intel['match_score'])

        return jsonify({
            "score": intel['match_score'],
            "status": "Market Ready" if intel['match_score'] > 75 else "Gap Identified",
            "experience": intel['years_of_experience'],
            "current_role": intel['current_role'],
            "missing": intel['gap_analysis'],
            "graph": graph_json,
            "credits_remaining": credits_left(current_user)
        })
    except Exception as e:
        print(f"Upload Error: {str(e)}")
        if charged:
            refund_credits(current_user, COSTS['cv_analysis'])
        return jsonify({"error": "Analysis failed. Please try again."}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    init_db()
    app.run(debug=False, port=5001)