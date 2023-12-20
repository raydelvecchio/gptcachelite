# GPTCacheLite: Semantic OpenAI Query Caching System
Semantic cache wrapper around the OpenAI API. Powered by [Vlite V2](https://github.com/raydelvecchio/vlite-v2). Install with `pip install gptcachelite`. Can find the PyPi distribution [here](https://pypi.org/project/gptcachelite/).

# About
You never want to repeat LLM calls, especially if they'll result in the same thing. Save on time, API costs, and more with GPT Cache Lite! Inspired by the
original [GPTCache](https://github.com/zilliztech/GPTCache). This cache system supports wrappers for *both sync and async* OpenAI API calls, and is incredibly
lightweight compared to GPTCache. Powered entirely by Vlite V2 on the backend, achieve blazing fast caching and retrieval to speed up your inference.

# How it Works
1. You submit a query to the OpenAI API, just as you would normally.
2. gptcachelite checks your query to see if there's any rough semantic match to a query/response pair you've seen in the past.
3. If there's a match, we return the response seen with that semantically similar query previously, and no LLM is called.
4. If there's no match, get the response from the OpenAI API.
5. Cache this query / response pair for search in step 2 for future queries!

# Synchronous Example
```python
    from gptcachelite import OpenAICache
    import dotenv
    import os
    dotenv.load_dotenv('.env')

    cache = OpenAICache('test_cache.npz', openai_key=os.environ['OPENAI_API_KEY'])
    model="gpt-3.5-turbo"
    messages=[
        {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
        {"role": "user", "content": "What is a Llama?"}
    ]
    
    response = cache.complete(model, messages)
    print(response)
```

# Async Example
```python
    from gptcachelite import AsyncOpenAICache
    import dotenv
    import asyncio
    import os
    dotenv.load_dotenv('.env')

    async def main():
        cache = AsyncOpenAICache('test_cache.npz', openai_key=os.environ['OPENAI_API_KEY'])
        model="gpt-3.5-turbo"
        messages=[
            {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
            {"role": "user", "content": "What is a Llama?"}
        ]
        response = await cache.complete(model, messages)
        print(response)
    
    if __name__ == "__main__":
        asyncio.run(main())
```

# Pip Deploy
1. `python3 setup.py sdist bdist_wheel`
2. `twine upload dist/*`
