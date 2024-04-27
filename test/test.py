import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from gptcachelite import SemanticCache

def test():
    cache = SemanticCache(db_name="test", openai_key="nice try")
    tm1 = [{"role": "system", "content": "I am a robot"}, {"role": "user", "content": "Whats a robot"}]
    response = cache.complete(provider='openai', model='gpt-3.5-turbo', messages=tm1)
    print(response)
    tm2 = [{"role": "system", "content": "I am a robot"}, {"role": "user", "content": "name a robot!"}]
    response = cache.complete(provider='openai', model='gpt-3.5-turbo', messages=tm2)
    print(response)
    tm3 = [{"role": "system", "content": "I am a robot"}, {"role": "user", "content": "how are olives grown"}]
    response = cache.complete(provider='openai', model='gpt-3.5-turbo', messages=tm3)
    print(response)
    tm4 = [{"role": "system", "content": "I am a robot"}, {"role": "user", "content": "What is a robot?"}]
    response = cache.complete(provider='openai', model='gpt-3.5-turbo', messages=tm4)
    print(response)

if __name__ == "__main__":
    test()
