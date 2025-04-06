import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import os
import dotenv

# Load environment variables
dotenv.load_dotenv()

# Authenticate with Hugging Face
HUGGINGFACE_TOKEN = os.getenv('hfAccessToken')
login(HUGGINGFACE_TOKEN)

# Load the model (Microsoft Phi-4-mini-instruct)
MODEL_NAME = "microsoft/Phi-4-mini-instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_auth_token=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto", use_auth_token=True)


def extract_job_details(job_description):
    """Extracts structured job details using the Phi-4-mini-instruct model."""

    # Define a structured prompt
    prompt = f"""
    You are an AI job description parser. Extract the following details from the given job description:

    - Role Overview: Brief summary of the role.
    - Key Responsibilities: List the main responsibilities.
    - Qualifications & Skills: Required qualifications and skills.

    Job Description:
    {job_description}

    Output the extracted details in JSON format.
    """
    print("Start Tokenizing Job Details...")
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")
    print("Completed Tokenizing Job Details...")
    # Generate response
    print("Getting Job Description details...")
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.7)
    print("Completed Getting Job Details...")
    # Decode the output
    extracted_info = tokenizer.decode(output[0], skip_special_tokens=True)

    return extracted_info


# Example Job Description
job_description = """
Responsibilities

We are looking for a capable data scientist to join the Analytics team, reporting locally in India Bangalore. This person’s responsibilities include research, design and development of Machine Learning and Deep Learning algorithms to tackle a variety of Fraud-oriented challenges. The data scientist will work closely with software engineers and program managers to deliver end-to-end products, including: data collection in big scale and analysis, exploring different algorithmic approaches, model development, assessment and validation – all the way through production.

Qualifications

At least 3 years of hands-on development of complex Machine Learning models using modern frameworks and tools, ideally Python-based.
Solid understanding of statistics and applied mathematics
Creative thinker with a proven ability to tackle open problems and apply non-trivial solutions.
Experience in software development using Python, Java, or a similar language.
Any Graduate or M.Sc. in Computer Science, Mathematics, or equivalent, preferably in Machine Learning.
Ability to write clean and concise code
Quick learner, independent, methodical, and detail-oriented.
Team player, positive attitude, collaborative, good communication skills.
Dedicated, makes things happen.
Flexible, capable of making decisions in an ambiguous and changing environment.

Advantages:

Prior experience as a software developer or data engineer – advantage
Experience with Big data – advantage
Experience with Spark – big advantage
Experience with Deep Learning frameworks (PyTorch, TensorFlow, Keras) – advantage.
Experience in the Telecommunication domain and/or Fraud prevention - advantage
"""

# Extract details
output = extract_job_details(job_description)
print("Extracted Job Details:\n", output)
