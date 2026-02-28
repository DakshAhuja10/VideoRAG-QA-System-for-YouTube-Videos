#here we create a langchain document object which store the text,metadata
#this document object will then be used to create embeddings
#here we do not need to split the text (We do not use a seperate text splitter)simply because we #are generating the transcript line by line as can be seen in transcript_generate.py
# each line contains transript sentence from video ,start and end duration so the text is already #splitted  using a seperate text splitter can affect the start and end duration which may lead the #model hallucinate

import pandas as pd
from hashlib import sha256
from langchain_core.documents import Document

#we take the csv file as input and then compute the hash of the text column for every row 
#this allows us to detect duplicate content and prevents us from generating their embedding #repeatedly thereby save us cost by preventing us from making unnecessarily to google gemini api #for embeddings 
#this class returns a list of document object with each document object contianing page content as #text and all the remaining columns as meta data
#if a text is repeated multiple times still we create a document object simply because start and #end duration of those columns cannot be same


class Csv_Loader:
    
    def __init__(self,csv_path):
        self.csv_path=csv_path
        
    def _compute_text_hash(self,text:str) -> str :
        return sha256(text.encode("utf-8")).hexdigest()
    
    
    def load(self):
        df=pd.read_csv(self.csv_path)
        
        documents=[]
        
        for _,row in df.iterrows():
            text = row["text"]
            text_hash=self._compute_text_hash(text)
            
            # Metadata for citations & RAG
            metadata = {
                "video_id": row["video_id"],
                "title": row["title"],
                "start": row["start"],
                "duration": row["duration"],
                "url": row["url"],
                "publish_date": row.get("publish_date", None),
                "views": row.get("views", None),
                "length": row.get("length", None),
                "citation_url": f"{row['url']}&t={int(row['start'])}s",
                "text_hash":text_hash
            }
            
            documents.append(
                Document(
                    page_content=text,
                    metadata=metadata
                )
            )
        
        return documents

