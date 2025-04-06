from crewai import Agent, Task, Crew
from crewai.tools import ExtractorTool

# Define the extraction schema
job_schema = {
    "Role Overview": "Brief summary of the role.",
    "Key Responsibilities": "List of main responsibilities.",
    "Qualifications & Skills": "Required qualifications and skills."
}

# Create the ExtractorTool
extractor = ExtractorTool.from_schema(job_schema)

# Define the agent
job_parser_agent = Agent(
    role="Job Description Parser",
    goal="Extract structured information from job descriptions.",
    tools=[extractor],
    verbose=True
)

# Define the task
parse_job_task = Task(
    description="Extract key details (Role Overview, Responsibilities, Qualifications) from the job description.",
    agent=job_parser_agent
)

# Create the Crew
crew = Crew(
    agents=[job_parser_agent],
    tasks=[parse_job_task]
)

# Sample Job Description
job_description = """
We are looking for a Senior Software Engineer to join our AI team. The ideal candidate will be responsible for 
designing scalable machine learning solutions. 

Key Responsibilities:
- Develop and deploy AI-driven applications.
- Collaborate with cross-functional teams.
- Maintain and improve code quality.

Qualifications & Skills:
- Bachelor's degree in Computer Science or related field.
- Experience with Python, TensorFlow, and cloud platforms.
"""

# Run the agent
result = crew.kickoff(inputs={"input": job_description})
print(result)
