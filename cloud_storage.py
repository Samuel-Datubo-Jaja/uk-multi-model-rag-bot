import boto3
import os
import zipfile
from botocore.exceptions import ClientError
import streamlit as st

# def download_vectorstore():
#     """
#     Download vector store from S3 and extract it
#     """
#     try:
#         # Get AWS credentials from Streamlit secrets
#         aws_access_key = st.secrets["AWS_ACCESS_KEY_ID"]
#         aws_secret_key = st.secrets["AWS_SECRET_ACCESS_KEY"]
#         bucket_name = st.secrets["AWS_S3_BUCKET"]
        
#         # Create S3 client
#         s3 = boto3.client(
#             's3',
#             region_name='eu-west-2',
#             aws_access_key_id=aws_access_key,
#             aws_secret_access_key=aws_secret_key
#         )
        
#         # Download zip file
#         zip_path = "main_chroma_data.zip"
#         st.info("Downloading vector store from S3...")
#         s3.download_file(bucket_name, 'main_chroma_data.zip', zip_path)
        
#         # Extract zip file
#         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#             zip_ref.extractall("./")
        
#         # Remove zip file
#         os.remove(zip_path)
#         st.success("Vector store downloaded and extracted successfully!")
        
#     except ClientError as e:
#         st.error(f"Error downloading vector store: {str(e)}")
#         raise e
#     except Exception as e:
#         st.error(f"Unexpected error: {str(e)}")
#         raise e

def download_vectorstore():
    """Download vector store from S3 and extract it"""
    try:
        # Use direct URL approach
        url = "https://uk-building-regulations-vectorstore.s3.eu-west-2.amazonaws.com/main_chroma_data.zip"
        import requests
        
        st.info("Downloading vector store from S3...")
        zip_path = "main_chroma_data.zip"
        
        r = requests.get(url, stream=True)
        with open(zip_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("./")
        
        # Remove zip file
        os.remove(zip_path)
        st.success("Vector store downloaded and extracted successfully!")
        
    except Exception as e:
        st.error(f"Detailed error: {str(e)}")
        raise e


def upload_vectorstore():
    """
    Zip and upload vector store to S3
    Note: This is for your local use, not needed in the app
    """
    # Create zip file of main_chroma_data
    with zipfile.ZipFile('main_chroma_data.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('main_chroma_data'):
            for file in files:
                zipf.write(os.path.join(root, file))
    
    # Upload to S3
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name = os.getenv("AWS_S3_BUCKET")
    
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )
    
    s3.upload_file('main_chroma_data.zip', bucket_name, 'main_chroma_data.zip')
    print("Vector store uploaded successfully!")


# Add at the bottom of cloud_storage.py
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "upload":
        upload_vectorstore()  

#  run this on bash to upload vectore store
#  python cloud_storage.py upload              