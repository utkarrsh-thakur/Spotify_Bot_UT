import yaml
import spotipy.util as util

from langchain.requests import RequestsWrapper
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import *
from langchain import LLMChain
from langchain.tools.json.tool import JsonSpec

from langchain.output_parsers import *
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_toolkits import JsonToolkit


llm = ChatOpenAI(model_name = "gpt-3.5-turbo",temperature=0, verbose = True)

with open("spotify_openapi.yaml", encoding= 'utf-8') as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)
#json_spec = JsonSpec(dict_=raw_spotify_api_spec, max_value_length=6000)
#json_toolkit = JsonToolkit(spec=json_spec)

def construct_spotify_auth_headers(raw_spec: dict):
    scopes = list(raw_spec['components']['securitySchemes']['oauth_2_0']['flows']['authorizationCode']['scopes'].keys())
    access_token = util.prompt_for_user_token(scope=','.join(scopes))
    return {
        'Authorization': f'Bearer {access_token}'
    }

headers = construct_spotify_auth_headers(raw_spotify_api_spec)
requests_wrapper = RequestsWrapper(headers=headers)
spotify_agent = planner.create_openapi_agent(spotify_api_spec, requests_wrapper, llm=llm)


with open("openmeteo.yaml") as f:
    raw_open_meteo_api_spec = yaml.load(f, Loader=yaml.Loader)
meteo_api_spec = reduce_openapi_spec(raw_open_meteo_api_spec)
#json_spec1 = JsonSpec(dict_=raw_open_meteo_api_spec, max_value_length=6000)
#json_toolkit1 = JsonToolkit(spec=json_spec1)

requests_wrapper1 = RequestsWrapper()
open_meteo_agent = planner.create_openapi_agent(meteo_api_spec, requests_wrapper1, llm=llm)

tools = [
    Tool(
        name = "Open Spotify API",
        func=spotify_agent.run,
        description="useful when you need to answer questions about music. Input should be question giving full context",
    ),
    Tool(
        name="Open Meteo API",
        description="Useful for when you want to get weather forecast from the OpenMeteo API.",
        func=open_meteo_agent.run,
    ),
]


'''
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import BaseChatPromptTemplate
from langchain import SerpAPIWrapper, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, HumanMessage
import re

# Set up the base template
template = """Complete the objective as best you can. You have access to the following tools:

{tools}
Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

These were previous tasks you completed:

Begin!

Question: {input}
{agent_scratchpad}"""

# Set up a prompt template
class CustomPromptTemplate(BaseChatPromptTemplate):
    # The template to use
    template: str
    # The list of tools available
    tools: List[Tool]
    
    def format_messages(self, **kwargs) -> str:
        # Get the intermediate steps (AgentAction, Observation tuples)
        # Format them in a particular way
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts
        # Create a tools variable from the list of tools provided
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        # Create a list of tool names for the tools provided
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        formatted = self.template.format(**kwargs)
        return [HumanMessage(content=formatted)]
    
output_parser = CommaSeparatedListOutputParser()
prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    output_parser=output_parser,
    # This omits the `agent_scratchpad`, `tools`, and `tool_names` variables because those are generated dynamically
    # This includes the `intermediate_steps` variable because that is needed
    input_variables=["input", "intermediate_steps"]
)

agent1 = ReActTextWorldAgent(llm_chain=llm_chain,
                             stop=["\nObservation:"],
                             allowed_tools=tool_names)
'''


prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. 
            You have to strictly access only the following tools: 
            Only answer what is required and return the answer.
            If you cannot find the answer to given question, simply reply with I do not know the answer."""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input","agent_scratchpad"]
)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(
    llm_chain=llm_chain,
    allowed_tools=tool_names,
    stop={"\nObservation:"}
)

agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
query = "Which are the famous songs of Ariana Grande?"

agent_executor.run(query)

