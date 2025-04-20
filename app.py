import os
import time
import chromadb
from flask import Flask, request, jsonify, render_template
from docx import Document
from google import genai
from google.api_core import retry
from google.genai import types

app = Flask(__name__)

# Configure Google Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable not set")
client = genai.Client(api_key=GOOGLE_API_KEY)

# ChromaDB setup
clientdb = chromadb.PersistentClient(path="./chroma_db")
rent = clientdb.get_or_create_collection(name="rent_agreements")
nda = clientdb.get_or_create_collection(name="nda_agreements")
employment = clientdb.get_or_create_collection(name="employ_agreements")
franchise = clientdb.get_or_create_collection(name="franchise_agreements")
contractor = clientdb.get_or_create_collection(name="contractor_agreements")
all_dbs = [rent, nda, employment, franchise, contractor]

# Model configuration
model_config = types.GenerateContentConfig(temperature=0.75, top_p=0.9)

# Retry mechanism for API calls
def is_retriable(e):
    return isinstance(e, genai.errors.APIError) and e.code in {429, 503}

@retry.Retry(predicate=is_retriable)
def generate_embeddings(cl, etype):
    embedding_task = "retrieval_document" if etype else "retrieval_query"
    embed = client.models.embed_content(
        model="models/text-embedding-004",
        contents=cl,
        config=types.EmbedContentConfig(task_type=embedding_task)
    )
    return [e.values for e in embed.embeddings]

def read_docx(endname):
    path = f"./Clauses/{endname}.docx"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Document {path} not found")
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return paragraphs

def extract_doxc(path):
    doc = Document(path)
    return doc

def extract_samples(endname):
    dataset_path = f"./sampleagreements/{endname}/{endname}"
    if not os.path.exists(dataset_path):
        return []
    docx_files = []
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isfile(item_path) and item.lower().endswith('.docx'):
            docx_files.append(item_path)
    return docx_files

# Initialize ChromaDB with clauses
def initialize_chromadb():
    agreement_types = ["rent", "nda", "employment", "franchise", "contractor"]
    all_clauses = []
    for t in agreement_types:
        try:
            clauses = read_docx(t)
            all_clauses.append(clauses)
        except FileNotFoundError:
            all_clauses.append([])
            print(f"Warning: {t}.docx not found in Clauses/")
    
    for j, dataset in enumerate(all_clauses):
        if not dataset:
            continue
        embeds, ids, documents = [], [], []
        for i, clause in enumerate(dataset):
            vector = generate_embeddings(clause, True)
            time.sleep(0.4)  # Rate limit
            embeds.append(vector[0])
            ids.append(f"clause-{j}-{i}")
            documents.append(clause)
        all_dbs[j].add(embeddings=embeds, ids=ids, documents=documents)

# Run initialization if collections are empty
if not any(db.count() > 0 for db in all_dbs):
    initialize_chromadb()

def strip_type(agr: str):
    agreement_types = ["rent", "nda", "contractor", "employment", "franchise"]
    prompt = f"""Return the type of agreement in one lowercase word from {agreement_types}. Input: {agr}"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    return response.text.strip().lower()

def pos_neg(response: str):
    prompt = f"""Classify sentiment: Reply '1' for positive, '0' for negative. Sentence = {response}"""
    response_heat = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    return bool(int(response_heat.text))

def perform_analysis(atype, impt):
    prompt = f"""As a legal assistant, evaluate if {impt} is sufficient for a {atype} agreement. Respond with:
    - 'Yes. All essential information seems to be present.' if comprehensive.
    - 'No, The following essential information seems to be missing or unclear: [list]' if details are missing.
    - 'No, The provided information is too vague or insufficient.' if vague."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    return response.text, pos_neg(response.text)

def obtain_information_holes(final_type, important_info, extra_info):
    total_info = important_info + extra_info
    prompt = f"""Identify missing or unclear information for a {final_type} agreement based on: {total_info}. Return specific details needed."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    return response.text

def get_data(holes: str, final_type):
    prompt = f"""Retrieve information for {holes} to generate a {final_type} agreement."""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=model_config
    )
    return response.text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    global final_type, important_info, extra_info, req
    data = request.json
    user_input = data.get('message', '').strip()
    
    if not hasattr(app, 'state'):
        app.state = {
            'req': False,
            'final_type': None,
            'important_info': '',
            'extra_info': '',
            'step': 'agreement_type'
        }
    
    state = app.state
    
    if state['step'] == 'agreement_type':
        final_type = strip_type(user_input)
        agreement_types = ["rent", "nda", "contractor", "employment", "franchise"]
        if final_type not in agreement_types:
            return jsonify({'response': "Invalid agreement type. Please choose: rent, nda, contractor, employment, franchise"})
        state['final_type'] = final_type
        state['step'] = 'important_info'
        return jsonify({'response': f"Please provide important information for your {final_type} agreement (e.g., parties, duration, financial details)."})
    
    elif state['step'] == 'important_info':
        state['important_info'] = user_input
        state['step'] = 'extra_info'
        return jsonify({'response': f"Provide any extra information to tailor your {state['final_type']} agreement (e.g., specific clauses, conditions)."})
    
    elif state['step'] == 'extra_info':
        state['extra_info'] = user_input
        analysis, is_positive = perform_analysis(state['final_type'], state['important_info'])
        if not is_positive:
            state['step'] = 'important_info'
            return jsonify({'response': analysis + " Please provide more information."})
        
        state['req'] = True
        # Retrieve relevant clauses
        dbname = state['final_type'] + "_agreements"
        querydb = clientdb.get_collection(name=dbname)
        user_query = state['important_info'] + state['extra_info']
        query_embed = generate_embeddings(user_query, False)
        results = querydb.query(query_embeddings=query_embed, n_results=querydb.count())
        relevant_documents = results['documents']
        
        # Get sample agreements
        sample_agreements = extract_samples(state['final_type'])
        
        # Get additional info
        info_holes = obtain_information_holes(state['final_type'], state['important_info'], state['extra_info'])
        obtained_info = get_data(info_holes, state['final_type'])
        
        # Generate final agreement
        prompt = f"""Generate a {state['final_type']} agreement using:
        User info: {state['important_info']}
        Extra info: {state['extra_info']}
        Relevant clauses: {relevant_documents}
        Sample agreements: {sample_agreements}
        Additional info: {obtained_info}
        Ensure the agreement is concise, legally robust, and clear."""
        
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=model_config
        )
        
        # Reset state
        app.state = {'req': False, 'final_type': None, 'important_info': '', 'extra_info': '', 'step': 'agreement_type'}
        
        return jsonify({'response': response.text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.getenv('PORT', 5000)))