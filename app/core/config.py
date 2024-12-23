# app/core/config.py
import os
import boto3
from dotenv import load_dotenv

def setup_environment():
    # Load environment variables
    load_dotenv()

    # Set AWS defaults
    current_directory = os.getcwd()
    default_region_name = boto3.DEFAULT_SESSION.region_name if boto3.DEFAULT_SESSION else 'us-east-1'
    boto3.setup_default_session()
    current_profile_name = boto3.DEFAULT_SESSION.profile_name

    # Set environment defaults
    os.environ.setdefault('save_folder', current_directory)
    os.environ.setdefault('profile_name', current_profile_name)
    os.environ.setdefault('region_name', default_region_name)
    os.environ.setdefault('max_tokens', '4096')

    return {
        'save_folder': os.getenv('save_folder'),
        'profile_name': os.getenv('profile_name'),
        'region_name': os.getenv('region_name'),
        'max_tokens': int(os.getenv('max_tokens'))
    }