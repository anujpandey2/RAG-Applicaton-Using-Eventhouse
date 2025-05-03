#!/usr/bin/env python
# coding: utf-8

# ## Notebook 1
# 
# New notebook

# In[1]:


# The command is not a standard IPython magic command. It is designed for use within Fabric notebooks only.
# %pip install PyMuPDF azure.kusto.data azure.kusto.ingest pandas openai tiktoken pymupdf4llm umap-learn langchain


# In[2]:


from pyspark.sql import SparkSession

# Create a SparkSession if you don't already have one
spark = SparkSession.builder \
    .appName("PandasToSpark") \
    .getOrCreate()


# In[3]:


import pymupdf # imports the pymupdf library
from notebookutils import mssparkutils
import fitz
import os
from openai import AzureOpenAI
import tiktoken
import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter
import uuid
import re
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# In[4]:


client = AzureOpenAI(
        api_key='',
        api_version="2024-10-21",
        azure_endpoint=''
    )


# In[ ]:


#Step 1 : extract text from PDF and also perform chunking using langchain 
def extract_markdown(path:str,file:str)-> str:
    metadata={}
    all_chunks = []
    # document_id = uuid.uuid4()
    markdown = pymupdf4llm.to_markdown(path)
    section_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Smaller chunks (was 1000)
        chunk_overlap=30,  # Less overlap (was 100)
        separators=["\n## ", "\n# ", "\n", ". ", "? ", "! ", ";", ":", " - ", ",", " "]
    ) 
    chunks = section_splitter.split_text(markdown)
    print(f"chunks : {len(chunks)}")
    for i,chunk in enumerate( chunks ):

        metadata = {
            # 'document_id': str(document_id),
            # 'chunk_id' :f"{document_id}-{i}",
            'total_chunks':len(chunks),
            'file_name':file 
        }
        chunk= chunk.replace("#","")
        chunk = chunk.replace("*","")
        chunk = chunk.replace("","")
        chunk = chunk.replace("\n","")
        chunk_row = {
            'content': chunk,
            **metadata  # Unpack metadata into the row
        }
        all_chunks.append(chunk_row)
    return all_chunks


# In[ ]:


#step 2: check tokens
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


# In[ ]:


#Step 3: function to generate embedding
def embed(query):
    deployment_name = "text-embedding-ada-002"
    response = client.embeddings.create(input=query, model=deployment_name)
    response = response.data[0].embedding
    return response


# In[57]:


# Step 4: create dataframe with all information
pdf_path = "abfss://3ce47e0b-d1ce-4f60-b613-1e6097654ae4@onelake.dfs.fabric.microsoft.com/5d326f0d-364e-4697-865f-449f0771fd88/Files/data"
_markdown=[]
files = mssparkutils.fs.ls(pdf_path)
row = []
for file in files:
    file_name = file.name
    # _text = extraxt_text(f"/lakehouse/default/Files/data/{file_name}")
    _markdown += extract_markdown(f"/lakehouse/default/Files/data/{file_name}",file_name)
    # markdown_token = num_tokens_from_string(_markdown,"cl100k_base")
    # text_token = num_tokens_from_string(_text,"cl100k_base")
    # _embedding = embed(_markdown)
    # file_name_embedding = embed(file_name.split('.pdf')[0])

    # row_dict = {
    #         "filename": file_name,
    #         # "text": _text,
    #         "markdown" : ''.join(str(x) for x in _markdown),
    #         "metadata" :metadata,
    #         # "text_token":text_token,
    #         "markdown_token":markdown_token,
    #         "embedding":_embedding.data[0].embedding,
    #         "file_name_embedding":file_name_embedding.data[0].embedding
    #         }


     
    
    

import pandas as pd
df = pd.DataFrame(_markdown)
df['embedding'] = df['content'].apply(embed)
# print(df)

#convert to spark
spark_df = spark.createDataFrame(df)


# In[ ]:


# step 5 : setup kusto connector
AAD_TENANT_ID = "605c4983-49ba-4fd9-9fe2-f2fb2531394a"
KUSTO_CLUSTER =  "https://trd-tqqb0u8e194dqe9uek.z5.kusto.fabric.microsoft.com"
KUSTO_DATABASE = "StreamData"
KUSTO_TABLE = "Resume"
kustoOptions = {"kustoCluster": KUSTO_CLUSTER, "kustoDatabase" :KUSTO_DATABASE, "kustoTable" : KUSTO_TABLE }
access_token = mssparkutils.credentials.getToken(KUSTO_CLUSTER)


# In[ ]:


#Step 6: Write data to a Kusto table
spark_df.write. \
format("com.microsoft.kusto.spark.synapse.datasource"). \
option("kustoCluster","https://trd-tqqb0u8e194dqe9uek.z5.kusto.fabric.microsoft.com"). \
option("kustoDatabase",kustoOptions["kustoDatabase"]). \
option("kustoTable", kustoOptions["kustoTable"]). \
option("accessToken", mssparkutils.credentials.getToken("https://trd-tqqb0u8e194dqe9uek.z5.kusto.fabric.microsoft.com")). \
option("tableCreateOptions", "CreateIfNotExist").\
mode("Append"). \
save()


# In[ ]:


#Step 7 : prepare question for similarity search and query eventhouse 
search_query = "Which city is the cleanest and has best Air quality index"
search_embed = embed(search_query)
searchedEmbedding = search_embed
KustoQuery = f"Resume | extend similarity = series_cosine_similarity(dynamic("+str(searchedEmbedding)+"), embedding) | top 4 by similarity desc"


# In[ ]:


#read from KQL
new_df = spark.read.\
        format("com.microsoft.kusto.spark.synapse.datasource"). \
        option("kustoCluster","https://trd-tqqb0u8e194dqe9uek.z5.kusto.fabric.microsoft.com"). \
        option("kustoDatabase",kustoOptions["kustoDatabase"]). \
        option("kustoQuery", query).\
        option("accessToken", mssparkutils.credentials.getToken("https://trd-tqqb0u8e194dqe9uek.z5.kusto.fabric.microsoft.com")).load()
        



# In[ ]:


display( new_df)


# In[ ]:


#step 8: function to call OpenAI GPT model
def Call_openAi(text):
    response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=text,
            temperature=0
    )
    return response.choices[0].message.content


# In[ ]:


#Step 9 : function to get cosine similarity from Eventhouse
def fetch_similarity_eventhouse(question:str):
    searchedEmbedding = embed(question)
    KustoQuery = f"Resume | extend similarity = series_cosine_similarity(dynamic("+str(searchedEmbedding)+"), embedding) | top 2 by similarity desc"
    KustoDf = spark.read.\
        format("com.microsoft.kusto.spark.synapse.datasource"). \
        option("kustoCluster","https://trd-tqqb0u8e194dqe9uek.z5.kusto.fabric.microsoft.com"). \
        option("kustoDatabase",kustoOptions["kustoDatabase"]). \
        option("kustoQuery", KustoQuery).\
        option("accessToken", mssparkutils.credentials.getToken("https://trd-tqqb0u8e194dqe9uek.z5.kusto.fabric.microsoft.com")).load()
    return KustoDf


# In[ ]:


#step 10 : Ask questions from GPT model
search_query = "Find the cleanest city which has best Air quality index"
Answer_from_eventhouse = fetch_similarity_eventhouse(search_query)

Answer = ""

for row in Answer_from_eventhouse.rdd.toLocalIterator():
    Answer= Answer+" " + row['file_name']
# find_file_name = Answer_from_eventhouse['']

prompt = 'Question: {}'.format(search_query) +'\n'+'Information: {}'.format(Answer)

messages = [{"role":"system","content":"you are help assistant answering questions from users and You will only give the file name from the information provided."},
{"role":"user","content":prompt }] 

result = Call_openAi(messages)
display(result)


# In[ ]:


# function to decompose vectors and create clusters

def visualize_document_clusters(df, n_clusters=4, method='umap'):
    # Extract embeddings
    embeddings = np.array(df['embedding'].tolist())
    
    # Perform clustering on the original embeddings
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)
    
    # Reduce dimensions for visualization
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    elif method.lower() == 'umap':
        reducer = umap.UMAP(n_components=2, random_state=42)
    
    reduced_embeddings = reducer.fit_transform(embeddings)
    
    # Create visualization
    plt.figure(figsize=(12, 10))
    
    # Plot points colored by cluster
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=clusters,
        cmap='tab10',
        s=100,
        alpha=0.8
    )
    
    # Add labels
    #for i, (x, y) in enumerate(reduced_embeddings):
        #plt.annotate(df['file_name'].iloc[i], (x, y), fontsize=8)
    
    # Add cluster centers
    centers_reduced = reducer.transform(kmeans.cluster_centers_)
    plt.scatter(
        centers_reduced[:, 0],
        centers_reduced[:, 1],
        c='black',
        s=200,
        alpha=0.5,
        marker='X'
    )
    
    plt.title(f'Document Clusters Visualized with {method.upper()}')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Return DataFrame with cluster information
    result_df = df.copy()
    result_df['cluster'] = clusters
    return result_df


# In[66]:


visualize_document_clusters(df)


# In[58]:


#Step 1 (optional):  function to extract text from pdf using pymuhpdf library
def extraxt_text(path:str)-> str:
    text=''
    try:
        with fitz.open(path) as doc:
            for page in doc:
                text +=page.get_text()
                text = text.replace("\n", " ")
    except Exception as e:
        print(f'error from {e}')
    return text


# In[94]:


def chunk_resume_by_sections(resume_text):
    """Split resume into logical sections"""
    # Common resume section headers (expand as needed)
    section_patterns = [
        r"EDUCATION|ACADEMIC BACKGROUND",
        r"EXPERIENCE|WORK HISTORY|EMPLOYMENT",
        r"SKILLS|TECHNICAL SKILLS|COMPETENCIES",
        r"PROJECTS|PROJECT EXPERIENCE",
        r"CERTIFICATIONS|LICENSES",
        r"LANGUAGES|LANGUAGE PROFICIENCY"
    ]
    
    # Combine patterns
    combined_pattern = "|".join(section_patterns)
    
    # Find section headers
    import re
    headers = list(re.finditer(combined_pattern, resume_text, re.IGNORECASE))
    
    # Extract sections
    sections = []
    for i in range(len(headers)):
        start = headers[i].start()
        end = headers[i+1].start() if i < len(headers)-1 else len(resume_text)
        section_text = resume_text[start:end].strip()
        section_type = headers[i].group(0).upper()
        
        sections.append({
            "type": section_type,
            "content": section_text
        })
    #print(sections)
    return sections


# # Convert to DataFrame
# df = pd.DataFrame(chunked_resumes)


# In[99]:


#optional : Clean text
def clean_text_and_remove_categories(text):
    """Clean text and remove category headers"""
    
    # First, remove common markdown formatting
    text = text.replace("#", "").replace("*", "").replace("\n", "")
    
    # Common resume section headers to remove
    section_patterns = [
        r"EDUCATION|ACADEMIC BACKGROUND",
        r"EXPERIENCE|WORK HISTORY|EMPLOYMENT",
        r"SKILLS|TECHNICAL SKILLS|COMPETENCIES",
        r"PROJECTS|PROJECT EXPERIENCE",
        r"CERTIFICATIONS|LICENSES",
        r"LANGUAGES|LANGUAGE PROFICIENCY"
    ]
    
    # Combine patterns
    combined_pattern = "|".join(section_patterns)
    
    # Replace section headers with empty string
    text = re.sub(combined_pattern, "", text, flags=re.IGNORECASE)
    
    # Remove extra whitespace created by deletions
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# In[92]:


# optional : If chunking required as per sections
def extract_markdown(path:str,file:str)-> str:
    metadata={}
    all_chunks = []
    # document_id = uuid.uuid4()
    markdown = pymupdf4llm.to_markdown(path)
    chunked_resumes = []

    sections = chunk_resume_by_sections(markdown)
        
    for section in sections:
        chunked_resumes.append({
                "source": path,
                "section_type": section["type"],
                "content": section["content"]
            })
    print(f"chunks : {len(sections)}")
    
    return chunked_resumes


# In[28]:





# In[59]:





# In[64]:





# In[65]:




