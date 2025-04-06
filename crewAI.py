from crewai import Agent

##Create Job Description parser
jobParser=Agent(
    role="Detect Keypoints from Job description",
    goal = """
    You are an AI job description parser. Extract the following details from the given job description:
    - **Role Overview**: Brief summary of the role.
    - **Key Responsibilities**: List the main responsibilities.
    - **Qualifications & Skills**: Required qualifications and skills.
    Job Description:
    Output the extracted details in JSON format.
    """,
    verbose=True,
    memory=True,
    backstory=(
        "Expert in reading a Job description and Extracting keypoints such as Role Overview, Key Responsibilities, Qualifications & Skills",
    ),
    tools=[],
    allow_delegation=True,
)
