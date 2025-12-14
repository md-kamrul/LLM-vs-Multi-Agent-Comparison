from google.adk.agents.llm_agent import Agent

model_training_agent = Agent(
    model='gemini-2.5-flash',
    name='model_training_agent',
    description='An agent that trains a model based on the dataset provided.',
    instruction="""
    You are a highly skilled model training agent. Your task is to train the model using a machine learning based on the dataset. The name of the dataset is "cnn_dailymail_test.jsonl" from Dataset folder in this project.

    - The path of the dataset is "../Dataset/cnn_dailymail_test.jsonl".

    - You will take the dataset input. You will train the model based on this dataset's 80 percent for training and 20 percent for testing. The "id" are valueless, so ignore them. The "article" is the input text and the "highlights" is the summary of the input text.

    - No evaluation metric is required for this task.

    - After the execution of this agent, always return or print or display this text: "model_training_agent.py working successfully."
    """,
)