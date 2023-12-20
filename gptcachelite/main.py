from openai import AsyncOpenAI, OpenAI
from vlite2 import VLite
import numpy as np

class OpenAICache:
    def __init__(self, db_name: str, openai_key: str, auto_flush_amount: int = 50) -> None:
        self.db = VLite(collection_name=db_name)
        self.client = OpenAI(api_key=openai_key)
        self.flush_amount = auto_flush_amount

    def complete(self, model: str, messages: list[dict[str, str]], cache: bool = True, pull_cache: bool = True, threshold: float = 0.7) -> str:
        """
        Wrapper of the OpenAI completions.create API. Takes in the same model and messages, with additinoal parameters on whether to cache, pull from cache,
        and the similarity threshold to pull from cache.
        """
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")
            if 'role' not in message or 'content' not in message:
                raise ValueError("Each message must contain 'role' and 'content' keys.")
            if message['role'].lower() not in ['system', 'user', 'assistant']:
                raise ValueError("The 'role' key must be either 'system' or 'user'.")

        query = messages[-1]['content']

        if pull_cache:
            _, metadata, sims = self.db.remember(text=query, top_k=1, autocut=False, return_metadata=True, return_similarities=True)
            if metadata and sims:
                if sims[0] > threshold:
                    return metadata[0]['cached_response']
            
        completion = self.client.chat.completions.create(model=model, messages=messages)
        response = completion.choices[0].message.content
        
        if cache and query not in self.db.texts:
            if len(self.db.texts) >= self.flush_amount:
                print("Flushing out database.")
                self.flush()
            self.db.memorize(query, max_seq_length=128, metadata={'cached_response': response})  # storing the response in the metadata field

        return response
    
    def flush(self):
        """
        Flushes the cache by clearing our database of cached response / query pairs.
        """
        self.db.texts = []
        self.db.metadata = {}
        self.db.vectors = np.empty((0, self.db.model.dimension))


class AsyncOpenAICache:
    def __init__(self, db_name: str, openai_key: str, auto_flush_amount: int = 50) -> None:
        self.db = VLite(collection_name=db_name)
        self.client = AsyncOpenAI(api_key=openai_key)
        self.flush_amount = auto_flush_amount

    async def complete(self, model: str, messages: list[dict[str, str]], cache: bool = True, pull_cache: bool = True, threshold: float = 0.7) -> str:
        """
        Wrapper of the OpenAI completions.create API. Takes in the same model and messages, with additinoal parameters on whether to cache, pull from cache,
        and the similarity threshold to pull from cache.
        """
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")
            if 'role' not in message or 'content' not in message:
                raise ValueError("Each message must contain 'role' and 'content' keys.")
            if message['role'].lower() not in ['system', 'user', 'assistant']:
                raise ValueError("The 'role' key must be either 'system' or 'user'.")

        query = messages[-1]['content']

        if pull_cache:
            _, metadata, sims = self.db.remember(text=query, top_k=1, autocut=False, return_metadata=True, return_similarities=True)
            if metadata and sims:
                if sims[0] > threshold:
                    return metadata[0]['cached_response']
            
        completion = await self.client.chat.completions.create(model=model, messages=messages)
        response = completion.choices[0].message.content
        
        if cache and query not in self.db.texts:
            if len(self.db.texts) >= self.flush_amount:
                print("Flushing out database.")
                self.flush()
            self.db.memorize(query, max_seq_length=128, metadata={'cached_response': response})  # storing the response in the metadata field

        return response
    
    def flush(self):
        """
        Flushes the cache by clearing our database of cached response / query pairs.
        """
        self.db.texts = []
        self.db.metadata = {}
        self.db.vectors = np.empty((0, self.db.model.dimension))
