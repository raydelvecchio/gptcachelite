# GPTCacheLite: Semantic LLM Query Caching System
Semantic cache wrapper around LLM APIs. Powered by [Vlite V2](https://github.com/raydelvecchio/vlite-v2). Install with `pip install gptcachelite`. Can find the PyPi distribution [here](https://pypi.org/project/gptcachelite/).

# About
You never want to repeat LLM calls, especially if they'll result in the same thing. Save on time, API costs, and more with GPTCacheLite! Inspired by the
original [GPTCache](https://github.com/zilliztech/GPTCache). This cache system supports wrappers for *both sync and async* LLM API calls, and is incredibly
lightweight compared to GPTCache. Powered entirely by Vlite V2 on the backend, achieve blazing fast caching and retrieval to speed up your inference.

# How it Works
1. You submit a query to an LLM API (currently OpenAI and Mistral), just as you would normally.
2. gptcachelite checks your query to see if there's any rough semantic match to a query/response pair you've seen in the past.
3. If there's a match, we return the response seen with that semantically similar query previously, and no LLM is called.
4. If there's no match, get the response from the appropriate LLM API.
5. Cache this query / response pair for search in step 2 for future queries!

# Notes
* You pass in any API key you want to use to the constructor. If you don't do it for a model, you can't call their completions!

# Synchronous Example
```python
    from gptcachelite import SemanticCache
    import dotenv
    import os
    dotenv.load_dotenv('.env')

    cache = SemanticCache('test_cache.npz', openai_key=os.environ['OPENAI_API_KEY'])
    model="gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": "What is a Llama?"}
    ]
    
    response = cache.complete(provider='openai', model=model, messages=messages)
    print(response)
```

# Async Example
```python
    from gptcachelite import AsyncSemanticCache
    import dotenv
    import asyncio
    import os
    dotenv.load_dotenv('.env')

    async def main():
        cache = AsyncSemanticCache('test_cache.npz', mistral_key=os.environ['MISTRAL_API_KEY'])
        model="gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "What is a Llama?"}
        ]
        response = await cache.complete(provider="mistral", model=model, messages=messages)
        print(response)
    
    if __name__ == "__main__":
        asyncio.run(main())
```

# Pip Deploy
1. `python3 setup.py sdist bdist_wheel`
2. `twine upload dist/*`
