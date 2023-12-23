from setuptools import setup, find_packages

setup(
    name='gptcachelite',
    version='0.0.3',
    author='Ray Del Vecchio',
    author_email='ray@cerebralvalley.ai',
    description='OpenAI API Wrapper with Semantic Caching via Vlite2. See more at https://github.com/raydelvecchio/gptcachelite.',
    packages=find_packages(),
    install_requires=[
        'openai',
        'vlite2',
        'python-dotenv'
    ],
)