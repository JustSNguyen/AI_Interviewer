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


def get_excluded_questions():
    excluded_questions = []

    with open("excluded_interview_questions.txt", 'r') as file:
        excluded_questions = [line.strip() for line in file]
    
    return excluded_questions


def get_interview_question(excluded_questions=[]):
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
            5. Should not be in the list of excluded questions: {excluded_questions}.
            Focus on practical knowledge and concepts that are relevant to real-world scenarios.''' 

    agent = Agent(model, result_type=Question, system_prompt=prompt)

    result = agent.run_sync(prompt)

    return result.data

if __name__ == '__main__':
    excluded_questions = get_excluded_questions()
    question_data = get_interview_question(excluded_questions)
    print(question_data.question)

