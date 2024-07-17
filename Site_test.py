#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install llama-index')
get_ipython().system('pip install python-dotenv')
get_ipython().system('pip install llama-index transformers torch accelerate')
get_ipython().system('pip install llama-index-llms-huggingface')
get_ipython().system('pip install chromadb')


# ## Set up local embedding LLM

# In[ ]:


get_ipython().run_line_magic('pip', 'install llama-index-embeddings-huggingface')
get_ipython().run_line_magic('pip', 'install llama-index-embeddings-instructor')
get_ipython().run_line_magic('pip', 'install sentence-transformers')


# In[22]:


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


# ## Document reader

# In[ ]:


get_ipython().system('pip install llama-index-readers-web')


# In[ ]:


from llama_index.readers.web import WholeSiteReader

# Initialize the scraper with a prefix URL and maximum depth
scraper = WholeSiteReader(
    prefix="https://www.highspeedtraining.co.uk", max_depth=10
)

# Start scraping from a base URL
documents = scraper.load_data(
    base_url="https://www.highspeedtraining.co.uk/hub/"
)  # Example base URL


# ## Vector store

# In[ ]:


get_ipython().system('pip install llama-index-llms-openai')


# In[23]:


import os
os.environ['TOKENIZERS_PARALLELISM'] = 'False'


# In[20]:


# Setup database
db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection("site")


# In[28]:


# Create index 
from llama_index.llms.openai import OpenAI
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Set up OpenAI LLM
llm = OpenAI(model="gpt-3.5-turbo")

index = VectorStoreIndex.from_documents(
    documents,
    embed_model=embed_model,
    storage_context=storage_context,
    llm=llm
)


# In[38]:


query_engine = index.as_query_engine()


# ## Query

# In[62]:


response = query_engine.query("What are the top tips for managing challenging behaviour")
print(response)


# ## Citations

# In[64]:


from llama_index.core.query_engine import CitationQueryEngine


# In[65]:


citation_query_engine = CitationQueryEngine.from_args(
    index,
    similarity_top_k=3,
    # here we can control how granular citation sources are, the default is 512
    citation_chunk_size=512,
)


# In[78]:


response = citation_query_engine.query("Who is the Level 3 Supervising Food Safety course for?")
print(response)


# In[ ]:


for i in range(len(response.source_nodes)):
    print(response.source_nodes[i].node.get_text())


# In[79]:


chat_engine = index.as_chat_engine()


# In[ ]:


chat_engine.chat_repl()

