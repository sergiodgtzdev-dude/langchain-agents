from dotenv import load_dotenv
from langchain_classic import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()

tools =[TavilySearch()]
llm = ChatOpenAI(model="gpt-4")
react_prompt = hub.pull("hwchase17/react")
agent = create_react_agent(
    llm = llm,
    tools = tools,
    prompt= react_prompt
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke(
    input={
        "input" : "Search for the results of the last 3 Paris Saint Germain matches and do a summary of the score"
    }
)

def main():
    print(react_prompt)

if __name__ == "main":
    main()