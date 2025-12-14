from google.adk.agents.llm_agent import Agent

reasoning_agent = Agent(
    model='gemini-2.5-flash',
    name='reasoning_agent',
    description='An agent that understand input texts or paragraphs',
    instruction="""
    You are a highly skilled reasoning agent. Your task is to understand the input texts or paragraphs from the user input. 
    
    - You will collect the input and understand the input texts or paragraphs from the user.

    - You will understand the context, main ideas, meaning and key points of the input text.

    - You will call the summary_generator_agent to generate summary based on the input text or paragraph.
    """,
)
