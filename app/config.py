import os
from dotenv import load_dotenv

# load dotenv
def load_config():
    load_dotenv()
    root_env = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
    if os.path.exists(root_env):
        load_dotenv(root_env)

# function to get groq api key
def get_groq_api():
    return os.getenv('GROQ_API_KEY')
