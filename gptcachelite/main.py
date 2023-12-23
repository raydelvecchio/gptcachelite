from openai import AsyncOpenAI, OpenAI
from vlite2 import VLite
import numpy as np

class OpenAICache:
    def __init__(self, db_name: str, openai_key: str, auto_flush_amount: int = 50) -> None:
        self.db = VLite(collection_name=db_name)
        self.client = OpenAI(api_key=openai_key)
        self.flush_amount = auto_flush_amount  # after this many cached q/r pairs, flush the database and restart!

    def complete(self, model: str, messages: list[dict[str, str]], cache_query: str = None, check_cache: bool = False, write_cache: bool = True, read_cache: bool = True, threshold: float = 0.8) -> str:
        """
        Wrapper of the OpenAI completions.create API. Takes in the same model and messages, with additional parameters on whether to cache, pull from cache,
        and the similarity threshold to pull from cache.

        If we pass in a cache_query, use this to index into and save to the cache, instead of the last message's context. Used when we have some sort of 
        system prompt with context as the last message and we don't want to use the entire thing to do semantic search.

        If check_cache = True, we simply check the cache for the presence of our cache_query. If the cache_query is indeed present, we don't pull from the
        cache or return anything, and simply return "".
        """
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")
            if 'role' not in message or 'content' not in message:
                raise ValueError("Each message must contain 'role' and 'content' keys.")
            if message['role'].lower() not in ['system', 'user', 'assistant']:
                raise ValueError("The 'role' key must be either 'system', 'assistant', or 'user'.")

        if not cache_query:
            cache_query = messages[-1]['content']

        if read_cache:
            _, metadata, sims = self.db.remember(text=cache_query, top_k=1, autocut=False, return_metadata=True, return_similarities=True)
            if metadata and sims:
                if sims[0] > threshold:
                    return metadata[0]['cached_response'] if not check_cache else ""
            
        completion = self.client.chat.completions.create(model=model, messages=messages)
        response = completion.choices[0].message.content
        
        if write_cache and cache_query not in self.db.texts:
            if len(self.db.texts) >= self.flush_amount:
                print("Flushing out database.")
                self.flush()
            self.db.memorize(cache_query, max_seq_length=128, metadata={'cached_response': response})  # storing the response in the metadata field

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

    async def complete(self, model: str, messages: list[dict[str, str]], cache_query: str = None, check_cache: bool = False, write_cache: bool = True, read_cache: bool = True, threshold: float = 0.8) -> str:
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")
            if 'role' not in message or 'content' not in message:
                raise ValueError("Each message must contain 'role' and 'content' keys.")
            if message['role'].lower() not in ['system', 'user', 'assistant']:
                raise ValueError("The 'role' key must be either 'system', 'assistant', or 'user'.")

        if not cache_query:
            cache_query = messages[-1]['content']

        if read_cache:
            _, metadata, sims = self.db.remember(text=cache_query, top_k=1, autocut=False, return_metadata=True, return_similarities=True)
            if metadata and sims:
                if sims[0] > threshold:
                    return metadata[0]['cached_response'] if not check_cache else ""
            
        completion = await self.client.chat.completions.create(model=model, messages=messages)
        response = completion.choices[0].message.content
        
        if write_cache and cache_query not in self.db.texts:
            if len(self.db.texts) >= self.flush_amount:
                print("Flushing out database.")
                self.flush()
            self.db.memorize(cache_query, max_seq_length=128, metadata={'cached_response': response})  # storing the response in the metadata field

        return response
    
    def flush(self):
        self.db.texts = []
        self.db.metadata = {}
        self.db.vectors = np.empty((0, self.db.model.dimension))
