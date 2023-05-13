import yaml
import spotipy.util as util

from langchain.requests import RequestsWrapper
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents.agent_toolkits.openapi import planner
from langchain.agents import Tool
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain import LLMChain
from langchain.tools.json.tool import JsonSpec

llm = ChatOpenAI(model_name = "gpt-3.5-turbo",temperature=0, verbose = True)

with open("spotify_openapi.yaml", encoding= 'utf-8') as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)
json_spec = JsonSpec(dict_=raw_spotify_api_spec, max_value_length=6000)

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
json_spec = JsonSpec(dict_=raw_open_meteo_api_spec, max_value_length=6000)

requests_wrapper1 = RequestsWrapper()
open_meteo_agent = planner.create_openapi_agent(meteo_api_spec, requests_wrapper1, llm=llm)

tools = [
    Tool(
        name = "Open Spotify API",
        func=spotify_agent.run,
        description="useful when you have to get information related to songs from the Spotify API.",
        return_direct=True
    ),
    Tool(
        name="Open Meteo API",
        description="Useful for when you want to get weather forecast from the OpenMeteo API.",
        func=open_meteo_agent.run,
        return_direct=True
    ),
]

prefix = """Answer the following questions as best you can, but speaking as a pirate might speak. You have access to the following tools:"""
suffix = """Begin! Remember to speak as a pirate when giving your final answer. Use lots of "Args"

Question: {input}
{agent_scratchpad}"""

prompt = ZeroShotAgent.create_prompt(
    tools, 
    prefix=prefix, 
    suffix=suffix, 
    input_variables=["input", "agent_scratchpad"]
)

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)
tool_names = [tool.name for tool in tools]
agent = ZeroShotAgent(
    llm_chain=llm_chain,
    stop=["\nObservation:"], 
    allowed_tools=tool_names
)
agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True)
agent_executor.run("Tell me a Drake song.")
