from dotenv import load_dotenv

load_dotenv()

from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langsmith import Client

from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableLambda

from react_search_agent_traditional.prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTION
from react_search_agent_traditional.schemas import AgentResponse

tools = [TavilySearch()]
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
client = Client()
# react_prompt = client.pull_prompt("hwchase17/react",dangerously_pull_public_prompt=True)

#react_prompt = 
output_parser=PydanticOutputParser(pydantic_object=AgentResponse)
# defining react agent


react_prompt_with_format_instruction=PromptTemplate(
    template=REACT_PROMPT_WITH_FORMAT_INSTRUCTION,
    input_variables={"input","agent_scratchpad","tool_names"}
).partial(format_instructions=output_parser.get_format_instructions())

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=react_prompt_with_format_instruction,
)
# orchestrator
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
chain = agent_executor


def main() -> None:
    print("Hello from react-search-agent-traditional!")

    result = chain.invoke(
        input={
            "input": "search for top anime using langchain on myanimelist and list its details",
        }
    )
    print(result)


if __name__ == "__main__":
    main()
