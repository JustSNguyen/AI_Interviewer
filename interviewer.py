import os

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')


class Question(BaseModel):
    question: str 
    expected_answer: str 

def get_interview_question():
    model = GeminiModel('gemini-1.5-flash', api_key=gemini_api_key)

    topics = [
        "Operating Systems",
        "Computer Networks",
        "Databases",
        "Real-life Scenarios",
        "Data Structures",
        "Algorithms",
        "System Design",
        "Software Development Principles",
        "Security",
        "Version Control",
        "Web Development",
        "Testing",
        "Concurrency and Multithreading"
    ]

    prompt = f'''You're an interviewer for a software engineer position at a tech company. 
            Generate ONE random theory question and expected answer for that question on the topic of {topics}. The question should:
            1. Be specific enough to test the candidate's knowledge.
            2. Not require coding or implementation but can be answered verbally.
            3. Reflect common interview questions used by tech companies.
            4. Avoid overly rare or niche questions.
            Focus on practical knowledge and concepts that are relevant to real-world scenarios. 
            Make sure to provide a unique and varied question each time, even for the same topic.'''

    agent = Agent(model, result_type=Question, system_prompt=prompt)

    result = agent.run_sync(prompt)

    return result.data

question_data = get_interview_question()
print(question_data.question)

