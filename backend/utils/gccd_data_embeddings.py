import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from tqdm.auto import tqdm
import os
import re
import json

# Load the dataset
df = pd.read_csv('./dataset/gccd_event_details.csv')

# Initialize the model
model = SentenceTransformer('sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja')

# Function to combine features and normalize
def combine_and_normalize(row):
    features = [
        str(row['Start_Time']),
        str(row['End_Time']),
        str(row['Title']),
        str(row['Owner'])
    ]
    return ' | '.join(filter(lambda x: 'nan' not in x.lower(), features))

# Combine features
df['combined_features'] = df.apply(combine_and_normalize, axis=1)

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

# Create or connect to an index
index_name = "gccd-pune-event"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

# Get the index
index = pc.Index(index_name)

# Improved function to handle NaN values and clean strings
def handle_nan_and_clean(obj):
    if isinstance(obj, dict):
        return {k: handle_nan_and_clean(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [handle_nan_and_clean(v) for v in obj]
    elif pd.isna(obj) or (isinstance(obj, float) and np.isnan(obj)):
        return "Not Specified"  # Default value for NaN
    elif isinstance(obj, str):
        # Remove any non-printable characters and strip whitespace
        return ''.join(filter(lambda x: x.isprintable(), obj)).strip()
    else:
        return obj

# Function to validate metadata
def validate_metadata(metadata):
    return {k: (v if v is not None and v != "" else "Not Specified") for k, v in metadata.items()}

# Embed and upload data in batches
batch_size = 50

for i in tqdm(range(0, len(df), batch_size)):
    try:
        # Get a batch of data
        batch = df.iloc[i:i+batch_size]
        
        # Embed the combined features
        embeddings = model.encode(batch['combined_features'].tolist())
        
        # Prepare the batch for Pinecone
        ids = batch.index.astype(str).tolist()
        vectors = embeddings.tolist()
        
        # Prepare metadata
        metadata = batch[['Start_Time', 'End_Time', 'Title', 'Owner']].to_dict('records')
        
        # Handle empty values and validate metadata
        for j, meta in enumerate(metadata):
            meta = handle_nan_and_clean(meta)
            meta = validate_metadata(meta)
            metadata[j] = meta  # Update the metadata list with the processed dictionary

        # Upsert to Pinecone
        to_upsert = list(zip(ids, vectors, metadata))
        index.upsert(vectors=to_upsert)
        
    except Exception as e:
        print(f"Error during upsert for batch starting at index {i}: {e}")
        print(f"Problematic metadata: {json.dumps(metadata, indent=2, default=str)}")
        continue  # Skip this batch and continue with the next

print("Data embedding and storage complete!")