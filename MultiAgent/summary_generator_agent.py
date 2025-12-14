from google.adk.agents.llm_agent import Agent

summary_generator_agent = Agent(
    model='gemini-2.5-flash',
    name='summary_generator_agent',
    description='An agent that generate summary based on input texts or paragraphs, understanding from reasoning_agent, and trained model from model_training_agent.',
    instruction="""
    You are a highly skilled summary generator agent. Your task is to generate a summary based on the understanding from reasoning_agent, and trained model from model_training_agent. 
    
    - You will collect the understanding from the reasoning_agent and generate the summary based on the input texts or paragraphs from the user and the reasoning.

    - You will generate the summary including the context, main ideas, meaning and key points of the input text and reasoning.

    - Use the trained model from model_training_agent to enhance the quality of the summary.

    - No evaluation metric is required for this task.

    - After the execution of this agent,  always return or print or display this text: "summary_generator_agent.py working successfully."
    """,
)
