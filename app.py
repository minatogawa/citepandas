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

# Load environment variables from .env file
load_dotenv(override=True)

app = Flask(__name__)

# Set a default DATABASE_URI if it's not in the environment
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URI') or 'sqlite:///scopus_data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY') or 'fallback_secret_key'
app.config['CACHE_TYPE'] = 'simple'

db = SQLAlchemy(app)
cache = Cache(app)

# Set up OpenAI client
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

class Publication(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    authors = db.Column(db.String(500))
    author_full_names = db.Column(db.String(500))
    author_ids = db.Column(db.String(500))
    title = db.Column(db.String(500))
    year = db.Column(db.Integer)
    source_title = db.Column(db.String(200))
    volume = db.Column(db.String(50))
    issue = db.Column(db.String(50))
    art_no = db.Column(db.String(50))
    page_start = db.Column(db.String(50))
    page_end = db.Column(db.String(50))
    page_count = db.Column(db.Integer)
    cited_by = db.Column(db.Integer)
    doi = db.Column(db.String(100))
    link = db.Column(db.String(500))
    affiliations = db.Column(db.Text)
    authors_with_affiliations = db.Column(db.Text)
    abstract = db.Column(db.Text)
    author_keywords = db.Column(db.String(500))
    index_keywords = db.Column(db.String(500))
    molecular_sequence_numbers = db.Column(db.String(500))
    chemicals_cas = db.Column(db.String(500))
    tradenames = db.Column(db.String(500))
    manufacturers = db.Column(db.String(500))
    funding_details = db.Column(db.String(500))
    funding_texts = db.Column(db.Text)
    references = db.Column(db.Text)
    correspondence_address = db.Column(db.String(500))
    editors = db.Column(db.String(500))
    publisher = db.Column(db.String(500))
    sponsors = db.Column(db.String(500))
    conference_name = db.Column(db.String(500))
    conference_date = db.Column(db.String(500))
    conference_location = db.Column(db.String(500))
    conference_code = db.Column(db.String(500))
    issn = db.Column(db.String(50))
    isbn = db.Column(db.String(50))
    coden = db.Column(db.String(50))
    pubmed_id = db.Column(db.String(50))
    language_of_original_document = db.Column(db.String(50))
    abbreviated_source_title = db.Column(db.String(50))
    document_type = db.Column(db.String(50))
    publication_stage = db.Column(db.String(50))
    open_access = db.Column(db.String(50))
    source = db.Column(db.String(50))
    eid = db.Column(db.String(50))

def process_csv(file):
    # Clear existing records
    db.session.query(Publication).delete()
    db.session.commit()

    # Process the new CSV file
    df = pd.read_csv(file)
    for _, row in df.iterrows():
        publication = Publication(
            authors=row['Authors'],
            author_full_names=row['Author full names'],
            author_ids=row['Author(s) ID'],
            title=row['Title'],
            year=int(row['Year']),
            source_title=row['Source title'],
            volume=row['Volume'],
            issue=row['Issue'],
            art_no=row['Art. No.'],
            page_start=row['Page start'],
            page_end=row['Page end'],
            page_count=int(row['Page count']) if pd.notna(row['Page count']) else None,
            cited_by=int(row['Cited by']) if pd.notna(row['Cited by']) else None,
            doi=row['DOI'],
            link=row['Link'],
            affiliations=row['Affiliations'],
            authors_with_affiliations=row['Authors with affiliations'],
            abstract=row['Abstract'],
            author_keywords=row['Author Keywords'],
            index_keywords=row['Index Keywords'],
            molecular_sequence_numbers=row['Molecular Sequence Numbers'],
            chemicals_cas=row['Chemicals/CAS'],
            tradenames=row['Tradenames'],
            manufacturers=row['Manufacturers'],
            funding_details=row['Funding Details'],
            funding_texts=row['Funding Texts'],
            references=row['References'],
            correspondence_address=row['Correspondence Address'],
            editors=row['Editors'],
            publisher=row['Publisher'],
            sponsors=row['Sponsors'],
            conference_name=row['Conference name'],
            conference_date=row['Conference date'],
            conference_location=row['Conference location'],
            conference_code=row['Conference code'],
            issn=row['ISSN'],
            isbn=row['ISBN'],
            coden=row['CODEN'],
            pubmed_id=row['PubMed ID'],
            language_of_original_document=row['Language of Original Document'],
            abbreviated_source_title=row['Abbreviated Source Title'],
            document_type=row['Document Type'],
            publication_stage=row['Publication Stage'],
            open_access=row['Open Access'],
            source=row['Source'],
            eid=row['EID']
        )
        db.session.add(publication)
    db.session.commit()

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def get_ai_analysis(prompt):
    api_key = os.environ.get('OPENAI_API_KEY')
    
    if not api_key:
        return "Error: OpenAI API key not found in environment variables."
    
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a scholarly assistant tasked with analyzing bibliometric data from Scopus. "
                        "Write a detailed paragraph in the style of a results section from an academic paper. "
                        "Your analysis should not only describe the graph but also provide insights, interpretations, "
                        "and implications of the data."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error in AI analysis: {str(e)}"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            return redirect(request.url)
        file = request.files['csv_file']
        if file.filename == '':
            return redirect(request.url)
        if file and file.filename.endswith('.csv'):
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

if __name__ == '__main__':
    app.run(debug=True)