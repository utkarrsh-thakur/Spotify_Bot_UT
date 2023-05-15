from langchain.agents import (Tool, AgentType, initialize_agent)
from langchain.chat_models import ChatOpenAI
from langchain.requests import RequestsWrapper
import yaml
from langchain.agents.agent_toolkits.openapi.spec import reduce_openapi_spec
from langchain.agents.agent_toolkits.openapi import planner
import spotipy.util as util
from langchain.tools.json.tool import JsonSpec

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.2, verbose=True)

with open("spotify_openapi.yaml", encoding= 'utf-8') as f:
    raw_spotify_api_spec = yaml.load(f, Loader=yaml.Loader)
spotify_api_spec = reduce_openapi_spec(raw_spotify_api_spec)
json_spec = JsonSpec(dict_=raw_spotify_api_spec, max_value_length=4000)


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

requests_wrapper1 = RequestsWrapper()
open_meteo_agent = planner.create_openapi_agent(meteo_api_spec, requests_wrapper1, llm=llm)

tools = [
    
    Tool(
        name="Open Meteo API",
        description="Useful for when you want to get weather forecast from the OpenMeteo API.",
        func=open_meteo_agent.run,
    )
]

agent = initialize_agent(llm=llm, tools=tools, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
query = "What is the temperature like in Germany?"

agent.run(query)