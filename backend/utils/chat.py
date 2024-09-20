import json
import re
import os
import torch
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
import google.generativeai as genai
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import warnings

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="sentence_transformers")

# Configure Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Configure Pinecone
PINECONE_API_KEY = load_dotenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = "us-east-1"

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index_name = "gccd-pune-event-2024"
index = pc.Index(index_name)

# Device setup (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and tokenizer setup
model_name = "sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja"
retriever = SentenceTransformer(model_name).to(device)

# File path for your prompts.json
file_path = "backend/inputs/prompts.json"

# In-memory storage for conversation history
conversation_history = {}

def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    context = data["context"]
    examples = data["examples"]
    formatted_examples = []
    
    for example in examples:
        formatted_examples.append({
            "input": example["human"],
            "output": example["assistant"]
        })
    
    return context, formatted_examples

def get_gemini_response(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        #safety settings
        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]

        # Generate content with safety settings
        response = model.generate_content(
            prompt,
            safety_settings=safety_settings,
            generation_config={
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
        )
        
        return response.text
    except Exception as e:
        print(f"Error in get_gemini_response: {e}")
        return "I'm having trouble right now. Can we try again?"

def format_response(response):
    def convert_table(match):
        lines = match.group(1).strip().split('\n')
        html_table = "<table style='border-collapse: collapse; width: 100%; font-size: 0.6em;'>"
        for i, line in enumerate(lines):
            if i == 1 and set(line.strip()) <= set('|-'):  # Skip separator line
                continue
            cells = [cell.strip() for cell in line.split('|') if cell.strip() != '']
            if not cells:  # Skip empty rows
                continue
            html_table += "<tr>"
            cell_tag = "th" if i == 0 else "td"
            for cell in cells:
                html_table += f"<{cell_tag} style='border: 1px solid #ddd; padding: 8px; text-align: left;'>{cell if cell else '&nbsp;'}</{cell_tag}>"
            html_table += "</tr>"
        html_table += "</table>"
        return html_table

    # Convert markdown tables to HTML
    response = re.sub(r'((?:\n\|.*)+)', lambda m: convert_table(m), response)

    # Convert numbered lists and bullet points
    response = re.sub(r'(?m)^(\d+)\.\s', r'<br>\1. ', response)
    response = re.sub(r'(?m)^•\s', r'<br>• ', response)

    # Add bold to headings and important phrases
    response = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', response)
    
    # Replace double line breaks first
    response = response.replace('\n\n', '<br><br>')

    # Add emphasis to key phrases
    response = re.sub(r'\*(.*?)\*', r'<em>\1</em>', response)

    # Handle specific formatting for time and speaker roles
    response = re.sub(r'(\d{1,2}:\d{2} [AP]M)', r'<strong>\1</strong>', response)
    response = re.sub(r'(@\w+)', r'<em>\1</em>', response)


    # Replace single line breaks after replacing double line breaks
    response = response.replace('\n', '<br>')

    # Remove any leading/trailing whitespace and extra new lines
    response = response.strip()
    response = re.sub(r'^(<br>)+|(<br>)+$', '', response)

    # Remove extra spaces between HTML tags
    response = re.sub(r'>\s+<', '><', response)

    # Remove extra spaces after new lines
    response = re.sub(r'<br>\s+', '<br>', response)

    return response.strip()

def clean_response(response):
    # Remove any "Human:" or "Assistant:" prefixes
    response = re.sub(r'^(Human:|Assistant:)\s*', '', response, flags=re.IGNORECASE)
    # Remove any numbered list prefixes
    response = re.sub(r'^\d+\)\s*', '', response)
    # Remove any "Let's approach this step-by-step:" prefix
    response = re.sub(r'^Let\'s approach this step-by-step:\s*', '', response, flags=re.IGNORECASE)
    return response.strip()

def get_relevant_events(query, top_k=5):
    query_embedding = retriever.encode([query])[0].tolist()
    results = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True
    )
    
    relevant_events = []
    for match in results['matches']:
        event_info = match['metadata'].copy()  # Create a copy of the metadata
        event_info['score'] = float(match['score'])  # Ensure score is a float
        
        # Convert all values to strings to avoid type issues
        for key, value in event_info.items():
            event_info[key] = str(value)
        
        relevant_events.append(event_info)
    
    return relevant_events

def is_event_related(query):
    event_keywords = ['event', 'conference', 'meeting', 'workshop', 'seminar', 'date', 'time', 'location', 'schedule']
    return any(keyword in query.lower() for keyword in event_keywords)

def format_event_info(events):
    formatted_info = []
    for event in events[:3]:  # Limit to top 3 events for conciseness
        info = f"{event.get('Title', 'N/A')} (Start: {event.get('Start_Time', 'N/A')}, " \
               f"End: {event.get('End_Time', 'N/A')}, " \
               f"Owner: {event.get('Owner', 'N/A')}, " \
               f"Designation: {event.get('Designation', 'N/A')})"
        formatted_info.append(info)
    
    return ". ".join(formatted_info)

def generate_response(session_id, human_prompt):
    context, examples = read_json_file(file_path)
    
    example_prompt = PromptTemplate(
        input_variables=["input", "output"],
        template="Human: {input}\nAssistant: {output}"
    )

    few_shot_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        suffix=f"{context}\nCurrent conversation:\n{{history}}\nHuman: {{input}}\nAssistant:",
        input_variables=["input", "history"]
    )

    history = get_conversation_history(session_id)
    
    # Check if the query is event-related and retrieve relevant events
    if is_event_related(human_prompt):
        relevant_events = get_relevant_events(human_prompt)
        events_context = "Relevant events:\n" + format_event_info(relevant_events)
    else:
        events_context = ""
    
    prompt = few_shot_template.format(input=human_prompt, history=history) + "\n" + events_context
    
    response = get_gemini_response(prompt)
    response = clean_response(response)
    formatted_response = format_response(response)
    update_conversation_history(session_id, "Human", human_prompt)
    update_conversation_history(session_id, "Assistant", response)
    
    response_data = {
        "response": formatted_response,
    }
    return response_data

def get_conversation_history(session_id):
    if session_id not in conversation_history:
        return ""
    formatted_history = ""
    for entry in conversation_history[session_id]:
        formatted_history += f"{entry['sender']}: {entry['message']}\n"
    return formatted_history.strip()

def update_conversation_history(session_id, sender, message):
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    conversation_history[session_id].append({"sender": sender, "message": message})
    # Keep only the last 5 exchanges to maintain context without overwhelming the model
    conversation_history[session_id] = conversation_history[session_id][-10:]