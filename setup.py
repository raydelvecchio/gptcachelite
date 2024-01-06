from setuptools import setup, find_packages

setup(
    name='gptcachelite',
    version='0.0.6',
    author='Ray Del Vecchio',
    author_email='ray@cerebralvalley.ai',
    description='LLM (OpenAI and Mistral) API Wrapper with Semantic Caching via Vlite2. See more at https://github.com/raydelvecchio/gptcachelite.',
    packages=find_packages(),
    install_requires=[
        'openai',
        'vlite2',
        'python-dotenv',
        'mistralai'
    ],
)