from flask import Flask, render_template, request, redirect, url_for, jsonify, session
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
from flask_migrate import Migrate

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

# Verificar que tenemos la URI de PostgreSQL
postgres_uri = os.getenv('POSTGRES_URI')
if not postgres_uri:
    raise ValueError("POSTGRES_URI no está configurado en las variables de entorno")

print("Database URI:", postgres_uri)  # Para debugging

app = Flask(__name__)

# Configurar SQLAlchemy antes de criar la instancia
app.config['SQLALCHEMY_DATABASE_URI'] = postgres_uri
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', 'default-secret-key')
app.config['TINYMCE_API_KEY'] = os.getenv('TINYMCE_API_KEY', 'no-api-key')
app.config['FLASK_ENV'] = env

db = SQLAlchemy(app)
migrate = Migrate(app, db)
cache = Cache(app)

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

# Update the storage for rate limiting to use memory in development
if os.getenv('FLASK_ENV') == 'production':
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        storage_uri="memory://",
        default_limits=["200 per day", "50 per hour"]
    )
else:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["50 per day", "10 per hour"],
        storage_uri="memory://"
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
            model="gpt-4o-mini",
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

@app.route('/dados')
def dados():
    publications = Publication.query.all()
    return render_template('dados.html', publications=publications)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

# --- Funções de Análise de IA Separadas ---

@cache.memoize(timeout=36000)
def get_analysis_top_10_publishers():
    publications = Publication.query.all()
    if not publications:
        return "No publication data available for analysis."
    df = pd.DataFrame([(d.source_title, d.id) for d in publications], columns=['Publisher', 'ID'])
    top_10 = df['Publisher'].value_counts().nlargest(10)
    if top_10.empty:
        return "Not enough publisher data for analysis."
    prompt = f"Analyze this data of top 10 publishers: {top_10.to_string()}. Discuss the distribution of publications among these publishers, their market share, and any notable trends or dominance in the field based on this data."
    return get_ai_analysis(prompt)

@cache.memoize(timeout=36000)
def get_analysis_papers_per_year():
    publications = Publication.query.all()
    if not publications:
        return "No publication data available for analysis."
    df = pd.DataFrame([(d.year, d.id) for d in publications if d.year is not None], columns=['Year', 'ID'])
    if df.empty:
        return "Not enough yearly data for analysis."
    df_grouped = df.groupby('Year').count().reset_index()
    if df_grouped.empty:
        return "Not enough yearly data points for analysis."
    prompt = f"Analyze this data of papers published per year: {df_grouped.to_string()}. Discuss any notable patterns, trends, increases, or decreases in publication frequency over the years based on this data."
    return get_ai_analysis(prompt)

# Helper for streamgraph data extraction (to avoid duplication)
def _get_streamgraph_data():
    publications = Publication.query.all()
    if not publications:
        return None
    keyword_column = 'author_keywords'
    df = pd.DataFrame([(d.year, getattr(d, keyword_column)) for d in publications if d.year is not None], columns=['Year', 'Keywords'])

    def clean_keywords(keywords):
        if pd.isna(keywords) or keywords is None or keywords.strip() == '': return []
        return [k.strip().lower() for k in keywords.split(';') if k.strip().lower() not in ['', 'none']]

    df['Keywords'] = df['Keywords'].apply(clean_keywords)
    df = df.explode('Keywords').dropna(subset=['Keywords'])
    df = df[df['Keywords'] != '']
    if df.empty: return None

    keyword_counts = df['Keywords'].value_counts().reset_index()
    keyword_counts.columns = ['Keyword', 'counts']
    top_keywords = keyword_counts.nlargest(15, 'counts')['Keyword']
    if top_keywords.empty: return None

    df_top_keywords = df[df['Keywords'].isin(top_keywords)]
    df_keywords_count = df_top_keywords.groupby(['Year', 'Keywords']).size().reset_index(name='counts')
    return df_keywords_count, top_keywords


@cache.memoize(timeout=3600)
def get_analysis_keyword_streamgraph():
    stream_data = _get_streamgraph_data()
    if stream_data is None:
        return "Not enough keyword data for streamgraph analysis."
    df_keywords_count, _ = stream_data # We only need the count data for the prompt
    prompt = f"Analyze this data showing the frequency of the top 15 keywords over the years: {df_keywords_count.to_string()}. Discuss trends in keyword usage, emerging topics, declining topics, and any notable patterns based *only* on this provided data."
    return get_ai_analysis(prompt)

# --- Funções de Plot Atualizadas ---

@cache.memoize(timeout=36000)  # Cache for 10 hours
def create_plot_top_10_publishers():
    publications = Publication.query.all()
    df = pd.DataFrame([(d.source_title, d.id) for d in publications], columns=['Publisher', 'ID'])
    top_10 = df['Publisher'].value_counts().nlargest(10)
    if top_10.empty: # Handle case with no data for plot
        fig = go.Figure().update_layout(title='Top 10 Publishers (No data)')
        return json.dumps(fig.to_dict(), default=numpy_to_python)
        
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
    return plot_json # Only return plot json

@app.route('/graph/top_10_publishers')
@limiter.limit("2 per minute")  # Limite específico para esta rota
def graph_top_10_publishers():
    plot_json = create_plot_top_10_publishers()
    analysis = get_analysis_top_10_publishers() # New analysis function
    return render_template('graph.html', plot=plot_json, analysis=analysis, title='Top 10 Publishers')

@cache.memoize(timeout=36000)  # Cache for 10 hour
def create_plot_papers_per_year():
    publications = Publication.query.all()
    df = pd.DataFrame([(d.year, d.id) for d in publications if d.year is not None], columns=['Year', 'ID'])
    if df.empty:
        fig = go.Figure().update_layout(title='Papers Published per Year (No data)')
        return json.dumps(fig.to_dict(), default=numpy_to_python)
        
    df_grouped = df.groupby('Year').count().reset_index()
    if df_grouped.empty:
        fig = go.Figure().update_layout(title='Papers Published per Year (No data)')
        return json.dumps(fig.to_dict(), default=numpy_to_python)

    fig = px.bar(df_grouped, x='Year', y='ID', title='Papers Published per Year', labels={'ID': 'Number of Papers'})
    plot_json = json.dumps(fig.to_dict(), default=numpy_to_python)
    return plot_json # Only return plot json

@app.route('/graph/papers_per_year')
def graph_papers_per_year():
    plot_json = create_plot_papers_per_year()
    analysis = get_analysis_papers_per_year() # New analysis function
    return render_template('graph.html', plot=plot_json, analysis=analysis, title='Papers Published per Year')

@cache.memoize(timeout=3600)  # Cache for 1 hour
def create_streamgraph():
    stream_data = _get_streamgraph_data()
    if stream_data is None:
        fig = go.Figure().update_layout(title='Keyword Usage Over Time (No data)')
        return json.dumps(fig.to_dict(), default=numpy_to_python)

    df_keywords_count, top_keywords = stream_data

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
    return plot_json # Only return plot json

@app.route('/graph/keyword_streamgraph')
def graph_keyword_streamgraph():
    plot_json = create_streamgraph()
    analysis = get_analysis_keyword_streamgraph() # New analysis function
    return render_template('graph.html', plot=plot_json, analysis=analysis, title='Keyword Usage Over Time')

# --- Novas Rotas para Geração do Paper ---

@app.route('/generate_paper', methods=['POST'])
def generate_paper():
    try:
        document_items = request.json
        if not document_items:
            return jsonify({"error": "No items received"}), 400

        paper_structure = []
        analysis_cache = {} # Cache analyses within this request if needed
        graphs_to_render = [] # List to store plot data for the final page

        for item in document_items:
            item_type = item.get('type')
            item_name = item.get('name')
            element_info = f"Type: {item_type}, Name: {item_name}"

            if item_type == 'graph':
                analysis_text = "Analysis not available." # Default
                plot_json = None # Default plot data
                graph_name = item.get('name') # Use name to identify graph

                # Get analysis
                if graph_name == 'Top 10 Publishers':
                    if graph_name not in analysis_cache:
                         analysis_cache[graph_name] = get_analysis_top_10_publishers()
                    analysis_text = analysis_cache[graph_name]
                    plot_json = create_plot_top_10_publishers() # Get plot JSON
                elif graph_name == 'Papers Published per Year':
                     if graph_name not in analysis_cache:
                         analysis_cache[graph_name] = get_analysis_papers_per_year()
                     analysis_text = analysis_cache[graph_name]
                     plot_json = create_plot_papers_per_year() # Get plot JSON
                elif graph_name == 'Keyword Usage Over Time':
                     if graph_name not in analysis_cache:
                         analysis_cache[graph_name] = get_analysis_keyword_streamgraph()
                     analysis_text = analysis_cache[graph_name]
                     plot_json = create_streamgraph() # Get plot JSON
                
                element_info += f"\nAnalysis:\n{analysis_text}" # Add analysis to info for the prompt
                
                # Store graph data for rendering on the paper page
                if plot_json:
                    graphs_to_render.append({
                        "name": graph_name,
                        "plot_json": plot_json 
                    })

            paper_structure.append(f"[{element_info}]")

        # Construct the MORE RESTRICTIVE final prompt for OpenAI
        final_prompt = (
            "You are an assembly assistant. You will be given a sequence of elements, each marked with a Type and Name. Some elements might include pre-written Analysis text." 
            "Your task is to assemble these elements into a single document STRICTLY following the given sequence. "
            "1. For elements with Type 'title', output the Name as a title."
            "2. For elements with Type 'subtitle', output the Name as a subtitle."
            "3. For elements with Type 'paragraph', output the Name as a simple paragraph text."
            "4. For elements with Type 'conclusion', output the Name as a concluding paragraph."
            "5. For elements with Type 'graph', output ONLY the provided 'Analysis' text exactly as given. DO NOT elaborate, add headers, or perform extra calculations related to the graph analysis."
            "6. For any other element Type, output its Name as plain text."
            "7. DO NOT add any extra sections, titles, introductions, conclusions, or formatting beyond assembling the provided elements in the specified order."
            "Output only the assembled document content.\n\n"
            "Structure:\n" + "\n".join(paper_structure)
        )
        
        # --- Chamada à OpenAI com max_tokens aumentado --- 
        generated_paper_content = "Error generating paper." # Default
        try:
            api_key = os.environ.get('OPENAI_API_KEY')
            if not api_key:
                 generated_paper_content = "Error: OpenAI API key not found."
            else:
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    # Using the user-updated model gpt-4o-mini
                    model="gpt-4o-mini", 
                    messages=[
                        # Adjusted system prompt to reflect assembly task
                        {"role": "system", "content": "You are an assembly assistant putting document parts together according to strict instructions."}, 
                        {"role": "user", "content": final_prompt}
                    ],
                    max_tokens=1500 # Keep increased tokens for potentially long assembled content
                )
                generated_paper_content = response.choices[0].message.content.strip()
        except Exception as e:
            app.logger.error(f"Error calling OpenAI for paper generation: {str(e)}")
            generated_paper_content = f"Error during AI paper generation: {str(e)}"
            
        # Save both text and graph data to session
        session['generated_paper_text'] = generated_paper_content
        session['generated_paper_graphs'] = graphs_to_render # Store graph data
        
        return jsonify({"success": True, "redirect_url": url_for('paper')})

    except Exception as e:
        app.logger.error(f"Error in /generate_paper: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/paper')
def paper():
    # Retrieve both text and graph data from session
    paper_content = session.pop('generated_paper_text', 'Error: Generated paper content not found.')
    graphs_data = session.pop('generated_paper_graphs', []) # Retrieve graph data
    
    # Pass raw content, graph data, and API key to the template
    return render_template('paper.html', 
                           paper_content=paper_content, 
                           graphs_data=graphs_data,
                           tinymce_api_key=app.config['TINYMCE_API_KEY']) # Passar a chave

# Add this after all your models are defined (after the Publication class)
def init_db():
    with app.app_context():
        try:
            # Primero verificar que podemos conectar
            connection = db.engine.connect()
            
            # Verificar que es PostgreSQL usando una consulta específica
            try:
                result = connection.execute(text("SELECT current_setting('server_version')"))
                version = result.scalar()
                print(f"Connected to PostgreSQL version: {version}")
            except Exception as e:
                raise Exception("No se pudo verificar la versión de PostgreSQL. ¿Estás usando SQLite en lugar de PostgreSQL?")
            finally:
                connection.close()
            
            # Si llegamos aquí, estamos conectados a PostgreSQL
            db.create_all()
            print("Tablas creadas exitosamente")
            
        except Exception as e:
            print(f"Error de inicialización de la base de datos: {str(e)}")
            raise

# Comando Flask CLI para inicializar o DB
@app.cli.command('init-db')
def init_db_command():
    """Clear existing data and create new tables."""
    init_db()
    print('Initialized the database.')

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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

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

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        # seu código atual
        app.logger.info('Iniciando upload do arquivo')
        # mais código
    except Exception as e:
        app.logger.error(f'Erro no upload: {str(e)}')
        return str(e), 500

# Al inicio del archivo, después de los imports
load_dotenv(override=True)  # Forzar la carga de .env

# Imprimir las variables de entorno (solo para debugging)
print("Database URI:", os.getenv('POSTGRES_URI'))

