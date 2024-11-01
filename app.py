from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_caching import Cache
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
from sqlalchemy.sql import text
import re
from sqlalchemy import inspect
from config.config import DevelopmentConfig, ProductionConfig
import logging
from logging.handlers import RotatingFileHandler
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# Load environment variables based on FLASK_ENV
env = os.getenv('FLASK_ENV', 'development')
env_file = f'.env.{env}'

# Load environment variables from specific .env file
if os.path.exists(env_file):
    load_dotenv(env_file, override=True)
    print(f"Loaded environment from {env_file}")
else:
    print(f"Warning: {env_file} not found, falling back to .env")
    load_dotenv(override=True)

app = Flask(__name__)

# Configure app based on environment
if os.getenv('FLASK_ENV') == 'production':
    app.config.from_object(ProductionConfig)
else:
    app.config.from_object(DevelopmentConfig)

db = SQLAlchemy(app)
cache = Cache(app)

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["50 per day", "10 per hour"],
    storage_uri="memory://"  # Para produção, use Redis: "redis://localhost:6379"
)

class Publication(db.Model):
    __tablename__ = 'publication'
    id = db.Column(db.Integer, primary_key=True)
    authors = db.Column(db.Text)
    author_full_names = db.Column(db.Text)
    author_ids = db.Column(db.Text)
    title = db.Column(db.Text)
    year = db.Column(db.Integer)
    source_title = db.Column(db.Text)
    volume = db.Column(db.Text)
    issue = db.Column(db.Text)
    art_no = db.Column(db.Text)
    page_start = db.Column(db.Text)
    page_end = db.Column(db.Text)
    page_count = db.Column(db.Integer)
    cited_by = db.Column(db.Integer)
    doi = db.Column(db.Text)
    link = db.Column(db.Text)
    affiliations = db.Column(db.Text)
    authors_with_affiliations = db.Column(db.Text)
    abstract = db.Column(db.Text)
    author_keywords = db.Column(db.Text)
    index_keywords = db.Column(db.Text)
    molecular_sequence_numbers = db.Column(db.Text)
    chemicals_cas = db.Column(db.Text)
    tradenames = db.Column(db.Text)
    manufacturers = db.Column(db.Text)
    funding_details = db.Column(db.Text)
    funding_texts = db.Column(db.Text)
    references = db.Column(db.Text)
    correspondence_address = db.Column(db.Text)
    editors = db.Column(db.Text)
    publisher = db.Column(db.Text)
    sponsors = db.Column(db.Text)
    conference_name = db.Column(db.Text)
    conference_date = db.Column(db.Text)
    conference_location = db.Column(db.Text)
    conference_code = db.Column(db.Text)
    issn = db.Column(db.Text)
    isbn = db.Column(db.Text)
    coden = db.Column(db.Text)
    pubmed_id = db.Column(db.Text)
    language_of_original_document = db.Column(db.Text)
    abbreviated_source_title = db.Column(db.Text)
    document_type = db.Column(db.Text)
    publication_stage = db.Column(db.Text)
    open_access = db.Column(db.Text)
    source = db.Column(db.Text)
    eid = db.Column(db.Text)

def sanitize_string(value):
    if value is None or pd.isna(value):
        return None
    # Remove any non-alphanumeric characters except spaces, commas, periods, and hyphens
    return str(value).strip()

def sanitize_int(value):
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (ValueError, TypeError):
        return None

def process_csv(file):
    try:
        print("Starting CSV processing...")
        
        # Otimizar a deleção de registros para PostgreSQL
        try:
            db.session.execute(text('TRUNCATE TABLE publication RESTART IDENTITY CASCADE'))
            db.session.commit()
            print("Successfully truncated publication table")
        except Exception as e:
            db.session.rollback()
            print(f"Error truncating table: {str(e)}")
            raise

        # Ler o CSV
        try:
            df = pd.read_csv(file)
            print(f"Successfully read CSV file with {len(df)} rows")
            
            # Verificar as colunas do CSV
            print("CSV columns:", df.columns.tolist())
        except Exception as e:
            print(f"Error reading CSV file: {str(e)}")
            raise

        # Processar cada linha
        records_processed = 0
        for index, row in df.iterrows():
            try:
                publication = Publication(
                    authors=sanitize_string(row.get('Authors')),
                    author_full_names=sanitize_string(row.get('Author full names')),
                    author_ids=sanitize_string(row.get('Author(s) ID')),
                    title=sanitize_string(row.get('Title')),
                    year=sanitize_int(row.get('Year')),
                    source_title=sanitize_string(row.get('Source title')),
                    volume=sanitize_string(row.get('Volume')),
                    issue=sanitize_string(row.get('Issue')),
                    art_no=sanitize_string(row.get('Art. No.')),
                    page_start=sanitize_string(row.get('Page start')),
                    page_end=sanitize_string(row.get('Page end')),
                    page_count=sanitize_int(row.get('Page count')),
                    cited_by=sanitize_int(row.get('Cited by')),
                    doi=sanitize_string(row.get('DOI')),
                    link=sanitize_string(row.get('Link')),
                    affiliations=sanitize_string(row.get('Affiliations')),
                    authors_with_affiliations=sanitize_string(row.get('Authors with affiliations')),
                    abstract=sanitize_string(row.get('Abstract')),
                    author_keywords=sanitize_string(row.get('Author Keywords')),
                    index_keywords=sanitize_string(row.get('Index Keywords')),
                    molecular_sequence_numbers=sanitize_string(row.get('Molecular Sequence Numbers')),
                    chemicals_cas=sanitize_string(row.get('Chemicals/CAS')),
                    tradenames=sanitize_string(row.get('Tradenames')),
                    manufacturers=sanitize_string(row.get('Manufacturers')),
                    funding_details=sanitize_string(row.get('Funding Details')),
                    funding_texts=sanitize_string(row.get('Funding Texts')),
                    references=sanitize_string(row.get('References')),
                    correspondence_address=sanitize_string(row.get('Correspondence Address')),
                    editors=sanitize_string(row.get('Editors')),
                    publisher=sanitize_string(row.get('Publisher')),
                    sponsors=sanitize_string(row.get('Sponsors')),
                    conference_name=sanitize_string(row.get('Conference name')),
                    conference_date=sanitize_string(row.get('Conference date')),
                    conference_location=sanitize_string(row.get('Conference location')),
                    conference_code=sanitize_string(row.get('Conference code')),
                    issn=sanitize_string(row.get('ISSN')),
                    isbn=sanitize_string(row.get('ISBN')),
                    coden=sanitize_string(row.get('CODEN')),
                    pubmed_id=sanitize_string(row.get('PubMed ID')),
                    language_of_original_document=sanitize_string(row.get('Language of Original Document')),
                    abbreviated_source_title=sanitize_string(row.get('Abbreviated Source Title')),
                    document_type=sanitize_string(row.get('Document Type')),
                    publication_stage=sanitize_string(row.get('Publication Stage')),
                    open_access=sanitize_string(row.get('Open Access')),
                    source=sanitize_string(row.get('Source')),
                    eid=sanitize_string(row.get('EID'))
                )
                db.session.add(publication)
                records_processed += 1
                
                # Commit a cada 100 registros para evitar transações muito grandes
                if records_processed % 100 == 0:
                    db.session.commit()
                    print(f"Committed {records_processed} records")
                    
            except Exception as e:
                print(f"Error processing row {index}: {str(e)}")
                print(f"Row data: {row.to_dict()}")
                db.session.rollback()
                raise

        # Commit final records
        try:
            db.session.commit()
            print(f"Successfully processed all {records_processed} records")
            
            # Verify final count
            final_count = db.session.query(Publication).count()
            print(f"Final count in database: {final_count}")
            
        except Exception as e:
            db.session.rollback()
            print(f"Error in final commit: {str(e)}")
            raise

    except Exception as e:
        print(f"Fatal error in process_csv: {str(e)}")
        db.session.rollback()
        raise

    return True

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@limiter.limit("2 per minute")  # Ajuste este valor conforme sua necessidade
def get_ai_analysis(prompt):
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        return "Error: OpenAI API key not found in environment variables."
    
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a scholarly assistant..."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in AI analysis: {str(e)}"

# Initialize CSRF protection
csrf = CSRFProtect(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return redirect(request.url)
        file = request.files['csv_file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Process the file (e.g., save it temporarily or process directly)
            process_csv(file)
            cache.clear()  # Clear the cache when new data is uploaded
            return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    publications = Publication.query.all()
    return render_template('dashboard.html', publications=publications)

@cache.memoize(timeout=36000)  # Cache for 10 hours
def create_plot_top_10_publishers():
    publications = Publication.query.all()
    df = pd.DataFrame([(d.source_title, d.id) for d in publications], columns=['Publisher', 'ID'])
    top_10 = df['Publisher'].value_counts().nlargest(10)
    
    fig = px.pie(
        values=top_10.values,
        names=top_10.index,
        title='Top 10 Publishers',
        labels={'label': 'Publisher', 'value': 'Number of Publications'}
    )
    fig.update_traces(
        textposition='inside',
        textinfo='percent',
        hoverinfo='label+percent',
        textfont_size=12,
        insidetextorientation='radial'
    )
    fig.update_layout(
        legend_title_text='Publishers',
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.05
        )
    )
    plot_json = json.dumps(fig.to_dict(), default=numpy_to_python)
    
    # Generate AI analysis
    prompt = f"Analyze this pie chart of top 10 publishers: {top_10.to_string()}. Discuss the distribution of publications among these publishers, their market share, and any notable trends or dominance in the field."
    analysis = get_ai_analysis(prompt)
    
    return plot_json, analysis

@app.route('/graph/top_10_publishers')
@limiter.limit("2 per minute")  # Limite específico para esta rota
def graph_top_10_publishers():
    plot, analysis = create_plot_top_10_publishers()
    return render_template('graph.html', plot=plot, analysis=analysis, title='Top 10 Publishers')

@cache.memoize(timeout=36000)  # Cache for 10 hour
def create_plot_papers_per_year():
    publications = Publication.query.all()
    df = pd.DataFrame([(d.year, d.id) for d in publications], columns=['Year', 'ID'])
    df = df.groupby('Year').count().reset_index()
    fig = px.bar(df, x='Year', y='ID', title='Papers Published per Year')
    plot_json = json.dumps(fig.to_dict(), default=numpy_to_python)
    
    # Generate AI analysis
    prompt = f"Analyze this graph of papers published per year: {df.to_string()}. Discuss any notable patterns or changes in publication frequency."
    analysis = get_ai_analysis(prompt)
    
    return plot_json, analysis

@app.route('/graph/papers_per_year')
def graph_papers_per_year():
    plot, analysis = create_plot_papers_per_year()
    return render_template('graph.html', plot=plot, analysis=analysis, title='Papers Published per Year')

@cache.memoize(timeout=3600)  # Cache for 1 hour
def create_streamgraph():
    publications = Publication.query.all()
    
    # Print out the first publication's attributes
    # if publications:
    #     for attr, value in publications[0].__dict__.items():
    #         if not attr.startswith('_'):
    #             print(f"{attr}: {value}")

    # Use the correct column for keywords
    keyword_column = 'author_keywords'

    df = pd.DataFrame([(d.year, getattr(d, keyword_column)) for d in publications], columns=['Year', 'Keywords'])

    # Improved keyword normalization and filtering
    def clean_keywords(keywords):
        if pd.isna(keywords) or keywords is None or keywords.strip() == '':
            return []
        return [k.strip().lower() for k in keywords.split(';') if k.strip().lower() not in ['', 'none']]

    df['Keywords'] = df['Keywords'].apply(clean_keywords)
    df = df.explode('Keywords').dropna(subset=['Keywords'])
    df = df[df['Keywords'] != '']  # Remove any remaining empty strings

    # Count keyword frequencies
    keyword_counts = df['Keywords'].value_counts().reset_index()
    keyword_counts.columns = ['Keyword', 'counts']

    # Select top 15 keywords
    top_keywords = keyword_counts.nlargest(15, 'counts')['Keyword']
    df_top_keywords = df[df['Keywords'].isin(top_keywords)]
    
    # Count frequencies per year
    df_keywords_count = df_top_keywords.groupby(['Year', 'Keywords']).size().reset_index(name='counts')

    # Create the streamgraph
    fig = go.Figure()
    
    for keyword in top_keywords:
        keyword_data = df_keywords_count[df_keywords_count['Keywords'] == keyword]
        fig.add_trace(go.Scatter(
            x=keyword_data['Year'], 
            y=keyword_data['counts'],
            mode='lines',
            stackgroup='one',
            name=keyword
        ))
    
    fig.update_layout(
        title='Streamgraph of Top 15 Keywords Over the Years',
        xaxis_title='Year',
        yaxis_title='Keyword Frequency',
        legend_title='Keywords',
        hovermode='x unified'
    )
    
    plot_json = json.dumps(fig.to_dict(), default=numpy_to_python)

    # Generate AI analysis
    prompt = f"Analyze this streamgraph of top 15 keywords over the years: {df_keywords_count.to_string()}. Discuss trends in keyword usage, emerging topics, and any notable patterns."
    analysis = get_ai_analysis(prompt)
    
    return plot_json, analysis

@app.route('/graph/keyword_streamgraph')
def graph_keyword_streamgraph():
    plot, analysis = create_streamgraph()
    return render_template('graph.html', plot=plot, analysis=analysis, title='Keyword Usage Over Time')

# Add this after all your models are defined (after the Publication class)
def init_db():
    with app.app_context():
        try:
            # Verificar a string de conexão
            print(f"Attempting to connect with URI: {app.config['SQLALCHEMY_DATABASE_URI']}")
            
            # Testar conexão
            result = db.session.execute(text('SELECT 1'))
            print("Database connection test successful!")
            
            # Criar tabelas
            db.create_all()
            
            # Verificar tabelas
            inspector = inspect(db.engine)
            existing_tables = inspector.get_table_names()
            print(f"Available tables: {existing_tables}")
            
        except UnicodeDecodeError as e:
            print("Error: Database connection string contains invalid characters")
            print(f"Error details: {str(e)}")
            raise
        except Exception as e:
            print(f"Database initialization error: {str(e)}")
            raise

@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return render_template('500.html'), 500

# Add this at the bottom of the file, before running the app
if __name__ == '__main__':
    init_db()  # Create tables before running the app
    app.run(debug=True)

def setup_logging(app):
    if not os.path.exists('logs'):
        os.mkdir('logs')
    
    file_handler = RotatingFileHandler('logs/citepandas.log', maxBytes=10240, backupCount=10)
    
    if app.config['FLASK_ENV'] == 'production':
        file_handler.setLevel(logging.ERROR)
    else:
        file_handler.setLevel(logging.INFO)
        
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    )
    file_handler.setFormatter(formatter)
    app.logger.addHandler(file_handler)

# Adicione após criar a app
setup_logging(app)

