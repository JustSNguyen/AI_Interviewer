import os

from pydantic_ai import Agent
from pydantic_ai.models.gemini import GeminiModel
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

class Feedback(BaseModel):
    feedback: str
    rating: int
    should_review: bool
    follow_up_question: str = None 
    better_answer: str = None 

def get_feedback(original_question, candidate_answer):
    model = GeminiModel('gemini-1.5-flash', api_key=gemini_api_key)

    prompt = f'''You are an interviewer for a software engineer position at a tech company. 
            Review the following exchange:
            - Question: {original_question}
            - Candidate's Answer: {candidate_answer}
            Provide detailed feedback on the candidate's answer, including the following:
            1. A constructive evaluation highlighting what the candidate did well and areas they got wrong, missed, or could improve upon.
            2. A rating for the answer on a scale of 1 to 10, based on accuracy, completeness, and clarity.
            3. Whether the candidate should review this question again (answer "yes" or "no").
            4. A single follow-up question if applicable. If no follow-up question is needed, you don't need to provide this info
            5. If you can provide a better answer, include it here. Note that your answer MUST 
            continue the original answer and it should include information you laid out in your 
            feedback. You should try to provide an answer based on the vocabulary, tone of the 
            original answer.'''
    
    agent = Agent(model, result_type=Feedback, system_prompt=prompt)

    result = agent.run_sync(prompt)

    return result.data

if __name__ == '__main__':
    feedback_data = get_feedback(
        "What is the difference between a process and a thread?",
        "A process is an instance of a program that is being executed.  It has its own independent memory space, meaning that variables and data structures within one process are not directly accessible to other processes. A thread, on the other hand, is a smaller unit of execution within a process.  Multiple threads can exist within a single process and share the same memory space. This shared memory allows threads to communicate and share data easily but also requires careful management to avoid race conditions and data corruption.  Creating a new process involves significant overhead due to the need to allocate new memory and resources. Creating a new thread, however, involves much less overhead as it shares resources with its parent process.  Therefore, using threads can be more efficient for achieving concurrency when appropriate,  while processes offer better isolation and protection."
    )

    print(f"Feedback: {feedback_data.feedback}")
    print(f"Rating: {feedback_data.rating} / 10")
    print(f"Should review: {feedback_data.should_review}")
    print(f"Follow up question: {feedback_data.follow_up_question}")
    print(f"Better answer: {feedback_data.better_answer}")
    

