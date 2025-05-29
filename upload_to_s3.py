from cloud_storage import upload_vectorstore
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Verify environment variables are loaded
bucket_name = os.getenv("AWS_S3_BUCKET")
print(f"Bucket name: {bucket_name}")

if __name__ == "__main__":
    upload_vectorstore()