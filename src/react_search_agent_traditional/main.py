
from dotenv import load_dotenv
load_dotenv()

from langsmith import Client
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_core.prompts import PromptTemplate

tools=[TavilySearch()]
llm=ChatGoogleGenerativeAI(model='gemini-3.6-flash')
client = Client()
#react_prompt = client.pull_prompt("hwchase17/react",dangerously_pull_public_prompt=True)

react_prompt = PromptTemplate.from_template("""
Answer the following question as best you can. You have access to the following tools:

{tools}

You MUST follow this exact format.

Question: the input question you must answer
Thought: your reasoning about what to do
Action: one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action

Repeat Thought/Action/Action Input/Observation as necessary.

When you have enough information, you MUST finish with:

Thought: I now know the final answer
Final Answer: your answer to the original question

IMPORTANT:
- Do not write the final answer before "Final Answer:"
- Do not use Markdown headings instead of the required format.
- Do not output a normal answer until you reach "Final Answer:".
- When you are finished, the last section MUST begin with "Final Answer:".

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")

#defining react agent
agent=create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt,
)

#orchestrator
agent_executor=AgentExecutor(agent=agent,tools=tools, verbose=True)
chain=agent_executor


def main() -> None:
    print("Hello from react-search-agent-traditional!")
    

    result=chain.invoke(
        input={
            "input":"search for top 3 currently airing anime using langchain on myanimelist and list their details",
            }
        )
    print(result)


if __name__=="__main__":
    main()