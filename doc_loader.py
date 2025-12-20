import pandas as pd
from hashlib import sha256
from langchain.schema import Document


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

