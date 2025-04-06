from transformers import pipeline

# Load the text generation pipeline with the Mistral model
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-v0.3")

# Prompt the model to list South Asian countries
response = pipe("List of countries in South Asia:", max_length=100, do_sample=True)

# Print the generated response
print(response[0]['generated_text'])
