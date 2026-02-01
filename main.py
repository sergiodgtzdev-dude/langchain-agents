from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
import pprint
load_dotenv()

@tool
def search (query : str) -> str:
    """
    Tool that searches over the internet
    :param
        query: The query to search for
    :return:
        The search result
    """

    print(f"searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model="gpt-5")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from agents!")
    result = agent.invoke({"messages" : HumanMessage(content="Search for 3 job openings for AI engineer in Mexico City which are hybrid or remote and list their details")})
    pprint.pp(result)

if __name__ == "__main__":
    main()
