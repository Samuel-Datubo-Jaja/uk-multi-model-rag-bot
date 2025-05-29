# # Install the gradio-client package
# # pip install gradio-client

# from gradio_client import Client

# client = Client("https://samueljaja-llama3-1-8b-merged-uk-building-regulations.hf.space")
# result = client.predict(
#     "What are the fire safety requirements for commercial buildings?",  # prompt
#     0.7,  # temperature
#     512,  # max_tokens
#     0.9,  # top_p
#     api_name="/predict"  # Try with and without this parameter
# )
# print(result)

# import transformers
# print(f"Transformers version: {transformers.__version__}")

# import torch
# print(f"PyTorch version: {torch.__version__}")

# import sentence_transformers
# print(f"Sentence-Transformers version: {sentence_transformers.__version__}")

# import langchain_community
# print(f"LangChain Community version: {langchain_community.__version__}")


from gradio_client import Client

client = Client("SamuelJaja/llama3.1_8B_merged_uk_building_regulations")
result = client.predict(
		prompt="What are the minimum requirements for accessible bathroom facilities?",
		temp=0.1,
		max_tokens=256,
		top_p=0.9,
		api_name="/timed_predict_1"
)
print(result)