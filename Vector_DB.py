from transformers import AutoTokenizer, AutoModel
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
import weaviate
import json
import numpy as np
from weaviate.exceptions import UnexpectedStatusCodeException

   
class vector_db():    
    def __init__(self):
        self.embedding_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")
        self.client = weaviate.Client(
            url="https://nlp-app-zikua64p.weaviate.network/",  
            auth_client_secret=weaviate.auth.AuthApiKey(api_key="Dq8Im0L2BPA2iELXDVnlgRFZ4LYlFS8a0jgt"),  
        )

        class_name = "TranscriptSearch"
        self.class_obj = {"class": class_name, "vectorizer": "none"}

        self.client.schema.delete_class(class_name)
    #    self.client.schema.create_class(self.class_obj)


            

    def _class_exists(self, class_name):
        try:
            schema = self.client.schema.get()
            classes = [cls['class'] for cls in schema['classes']]
            return class_name in classes
        except UnexpectedStatusCodeException as e:
            print(f"An error occurred while checking if the class exists: {e}")
            return False

    def chunk_text(self, transcript):
        semantic_chunker = SemanticChunker(self.embedding_model, breakpoint_threshold_type="percentile")
        semantic_chunks = semantic_chunker.create_documents([transcript])
        list_of_chunks = [chunk.page_content for chunk in semantic_chunks]

        # Create a tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Falconsai/text_summarization")

        final_chunks = []
        for chunk in list_of_chunks:
            # Tokenize the chunk and split it into smaller chunks with a maximum of 500 tokens
            tokens = tokenizer(chunk, return_tensors='np', truncation=False).input_ids[0]
            sub_chunks = [tokens[i:i + 500] for i in range(0, len(tokens), 500)]
            
            # Convert tokens back to text and add to final chunks
            for sub_chunk in sub_chunks:
                final_chunks.append(tokenizer.decode(sub_chunk, skip_special_tokens=True))

        return final_chunks

    def create_vector_db(self, transcript):
        list_of_chunks = self.chunk_text(transcript)
        self.client.batch.configure(batch_size=len(list_of_chunks))
        with self.client.batch as batch:
            for i, doc in enumerate(list_of_chunks):
                properties = {"source_text": doc}
                vectors = self.embedding_model.embed_documents([doc])
                vector = np.array(vectors[0], dtype=np.float32).tolist()  # Flatten the first vector


                batch.add_data_object(properties, "TranscriptSearch", vector=vector)

    def query_vector_db(self, query):
        query_vector = self.embedding_model.embed_query(query)
        query_vector = np.array(query_vector, dtype=np.float32).tolist()


        result = self.client.query.get("TranscriptSearch", ["source_text"]).with_near_vector({"vector": query_vector, "certainty": 0.7}).with_limit(4).with_additional(['certainty', 'distance']).do()
        # Extract and concatenate source_text values
        source_texts = [item['source_text'] for item in result['data']['Get']['TranscriptSearch']]
        concatenated_source_text = "\n\n".join(source_texts)

        return concatenated_source_text
    
    def read_all_chunks(self):
        result = self.client.query.get("TranscriptSearch", ["source_text"]).with_limit(1000).do()
        chunks = [item['source_text'] for item in result['data']['Get']['TranscriptSearch']]

        return chunks


    def get_transcript(self, file_path):
        with open(file_path, 'r') as f:
            transcript = f.read()
        return transcript

        
   
        

        
