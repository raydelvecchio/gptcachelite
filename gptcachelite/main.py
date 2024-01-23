from openai import AsyncOpenAI, OpenAI
from mistralai.client import MistralClient
from mistralai.async_client import MistralAsyncClient
from mistralai.models.chat_completion import ChatMessage
from vlite2 import VLite2

class SemanticCache:
    def __init__(self, db_name: str, openai_key: str = "", mistral_key: str = "", auto_flush_amount: int = 50) -> None:
        self.openai_key = openai_key
        self.mistral_key = mistral_key
        self.db = VLite2(vdb_name=db_name)
        self.openai_client = OpenAI(api_key=openai_key)
        self.mistral_client = MistralClient(api_key=mistral_key)
        self.flush_amount = auto_flush_amount

    def __llm(self, provider: str, model: str, messages: list[dict[str, str]]) -> str:
        """
        Provides an LLM response given a provider, model, and messages.
        """
        if provider not in ['openai', 'mistral']:
            raise ValueError('provider must be one of [openai, mistral]')
        if provider == 'openai':
            if not self.openai_key:
                raise ValueError("OpenAI API key must be passed in to use OpenAI endpoints")
            completion = self.openai_client.chat.completions.create(model=model, messages=messages)
            return completion.choices[0].message.content
        if provider == 'mistral':
            if not self.mistral_key:
                raise ValueError("Mistral API key must be passed in to use Mistral endpoints")
            response = self.mistral_client.chat(model=model, messages=[ChatMessage(role=message['role'], content=message['content']) for message in messages])
            return response.choices[0].message.content

    def complete(self, provider: str, model: str, messages: list[dict[str, str]], cache_query: str = None, check_cache: bool = False, write_cache: bool = True, read_cache: bool = True, threshold: float = 0.8) -> str:
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
        if messages[0]['role'] not in ['system', 'user']:
            raise ValueError("The first role must be either 'user' or 'system'.")

        if not cache_query:
            cache_query = messages[-1]['content']

        if read_cache or check_cache:
            results = self.db.remember(text=cache_query, top_k=1, autocut=False, get_metadata=True, get_similarities=True)
            metadata = results['texts']
            sims = results['scores']
            if metadata and sims:
                if sims[0] > threshold:
                    return metadata[0]['cached_response'] if not check_cache else ""
            
        response = self.__llm(provider=provider, model=model, messages=messages)
        
        texts = self.db.get_texts().values()
        if write_cache and cache_query not in texts:
            if len(texts) >= self.flush_amount:
                print("Flushing out database.")
                self.flush()
            self.db.memorize(cache_query, max_seq_length=128, metadata={'cached_response': response})  # storing the response in the metadata field

        return response
    
    def flush(self):
        """
        Flushes the cache by clearing our database of cached response / query pairs.
        """
        self.db.clear()

class AsyncSemanticCache:
    def __init__(self, db_name: str, openai_key: str = "", mistral_key: str = "", auto_flush_amount: int = 50) -> None:
        self.openai_key = openai_key
        self.mistral_key = mistral_key
        self.db = VLite2(vdb_name=db_name)
        self.openai_client = AsyncOpenAI(api_key=openai_key)
        self.mistral_client = MistralAsyncClient(api_key=mistral_key)
        self.flush_amount = auto_flush_amount

    async def __llm(self, provider: str, model: str, messages: list[dict[str, str]]) -> str:
        """
        Provides an LLM response given a provider, model, and messages.
        """
        if provider not in ['openai', 'mistral']:
            raise ValueError('provider must be one of [openai, mistral]')
        if provider == 'openai':
            if not self.openai_key:
                raise ValueError("OpenAI API key must be passed in to use OpenAI endpoints")
            completion = await self.openai_client.chat.completions.create(model=model, messages=messages)
            return completion.choices[0].message.content
        if provider == 'mistral':
            if not self.mistral_key:
                raise ValueError("Mistral API key must be passed in to use Mistral endpoints")
            response = await self.mistral_client.chat(model=model, messages=[ChatMessage(role=message['role'], content=message['content']) for message in messages])
            return response.choices[0].message.content

    async def complete(self, provider: str, model: str, messages: list[dict[str, str]], cache_query: str = None, check_cache: bool = False, write_cache: bool = True, read_cache: bool = True, threshold: float = 0.8) -> str:
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Each message must be a dictionary.")
            if 'role' not in message or 'content' not in message:
                raise ValueError("Each message must contain 'role' and 'content' keys.")
            if message['role'].lower() not in ['system', 'user', 'assistant']:
                raise ValueError("The 'role' key must be either 'system', 'assistant', or 'user'.")
        if messages[0]['role'] not in ['system', 'user']:
            raise ValueError("The first role must be either 'user' or 'system'.")

        if not cache_query:
            cache_query = messages[-1]['content']

        if read_cache or check_cache:
            results = self.db.remember(text=cache_query, top_k=1, autocut=False, get_metadata=True, get_similarities=True)
            metadata = results['texts']
            sims = results['scores']
            if metadata and sims:
                if sims[0] > threshold:
                    return metadata[0]['cached_response'] if not check_cache else ""
            
        response = await self.__llm(provider=provider, model=model, messages=messages)
        
        texts = self.db.get_texts().values()
        if write_cache and cache_query not in texts:
            if len(texts) >= self.flush_amount:
                print("Flushing out database.")
                self.flush()
            self.db.memorize(cache_query, max_seq_length=128, metadata={'cached_response': response})  # storing the response in the metadata field

        return response
    
    def flush(self):
        self.db.clear()
