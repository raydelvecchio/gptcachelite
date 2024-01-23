import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from gptcachelite import SemanticCache

def test():
    cache = SemanticCache(db_name="test", openai_key="")
    test_messages = [{"role": "system", "content": "I am a robot"}, {"role": "user", "content": "What is a robot?"}]
    response = cache.complete(provider='openai', model='gpt-4', messages=test_messages)
    print(response)

if __name__ == "__main__":
    test()