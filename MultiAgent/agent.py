from google.adk.agents.llm_agent import Agent
from MultiAgent.reasoning_agent import reasoning_agent
from MultiAgent.model_training_agent import model_training_agent
from MultiAgent.summary_generator_agent import summary_generator_agent
from MultiAgent.proofreading_agent import proofreading_agent

root_agent = Agent(
    model='gemini-2.5-flash',
    name='summarization_agent_main',
    description='An agent that summarizes texts or paragraphs using multi-agents.',
    instruction="""
    You are a highly skilled summarization agent. Your task is to operate and manage all the agents related to the summarization process. 
    
    - You will take inputs from the user, which will be texts or paragraphs that need to be summarized.

    - You will delegate the summarization tasks to specialized sub-agents sequentially.

    - The multi-agent workflow: Take the input (agent.py) -> call the reasoning agent (reasoning_agent.py) -> train model from dataset (model_training_agent.py) -> summary generator agent (summary_generator_agent.py) -> proofreading agent (proofreading_agent.py) -> give the output

    - You will call the reasoning_agent and transfer the input texts or paragraphs to understand the input text.
    """,
    sub_agents=[reasoning_agent, model_training_agent, summary_generator_agent, proofreading_agent]
)
