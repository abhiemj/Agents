import requests
import json
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent
import os
from dotenv import load_dotenv
load_dotenv()

checkpointer = InMemorySaver()
LINKEDIN_BEARER_TOKEN = os.getenv("LINKEDIN_BEARER_TOKEN")
LINKEDIN_COOKIE = os.getenv("LINKEDIN_COOKIE")
LINKEDIN_UGC_POST_URL=os.getenv("LINKEDIN_UGC_POST_URL")
LINKEDIN_UGC_INFO = os.getenv("LINKEDIN_UGC_INFO")
LINKEDIN_UNIQUE_ID = os.getenv("LINKEDIN_UNIQUE_ID")


if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

if "SERPER_API_KEY" not in os.environ:
    os.environ["SERPER_API_KEY"] = os.environ.get("SERPER_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)




# url = LINKEDIN_UGC_INFO

# payload = {}
# headers = {
#   'Authorization': f'Bearer {LINKEDIN_BEARER_TOKEN}',
#   'Cookie': f'{LINKEDIN_COOKIE}',
# }

# response = requests.request("GET", url, headers=headers, data=payload)

# print(response.text)



serper = GoogleSerperAPIWrapper()

@tool("search_web_serper")
def search_web_serper(query: str) -> str:
    """
    Uses Serper API to search the web and return results as a string.
    """
    return serper.run(query)




@tool("post_content_on_linkedin")
def post_content_on_linkedin(content):
    """Function to post content on LinkedIn using UGC API."""
    url = LINKEDIN_UGC_POST_URL

    payload = json.dumps({
    "author": f"urn:li:person:{LINKEDIN_UNIQUE_ID}",
    "lifecycleState": "PUBLISHED",
    "specificContent": {
        "com.linkedin.ugc.ShareContent": {
        "shareCommentary": {
            "text": content
        },
        "shareMediaCategory": "NONE"
        }
    },
    "visibility": {
        "com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"
    }
    })
    headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {LINKEDIN_BEARER_TOKEN}',
    'Cookie': f'{LINKEDIN_COOKIE}'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.text




memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
tools = [post_content_on_linkedin,search_web_serper]  # Define your tools here

# 3. Bind memory to the agent
agent = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    memory=memory,
    verbose=True,
    agent_kwargs={
        "system_message": (
            "You are a helpful assistant that can search the web using search_web_serper tool and post content on LinkedIn using post_content_on_linkedin tool. Use the tools provided to assist with tasks. User will provide you a topic to post about on LinkedIn. You will search the web for relevant information , create a well structured post in bullet points and then post it on LinkedIn using the UGC API. You will also provide a summary of the post in a single line."
        )
    }
)

while True:
    try:
        user_input = input("You: ")
        result = agent.invoke({"input": user_input})
        print("Agent:", result["output"] if isinstance(result, dict) and "output" in result else result)
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"An error occurred: {str(e)}")




# config = {"configurable": {"thread_id": "1"}}
# while True:
#     query = input("Hi there! What would you like to do? (Type 'exit' to quit): ")
#     if query.lower() in ["exit", "quit", "stop"]:
#         print("Exiting the agent.")
#         break
#     user_input = {"messages": [{"role": "user", "content": query}]}
#     response = agent.stream(user_input,config=config)
#     # for chunk in agent.stream(user_input, config=config):
#         # print(chunk, end="", flush=True)
#     final_response = None
#     for step in agent.stream(user_input, config=config,debug=False,stream_mode="values"):
#         final_response = step  # The last yielded step will have the final state

#     # Extract the last AIMessage
#     ai_messages = [msg for msg in final_response['messages'] if msg.__class__.__name__ == "AIMessage"]
#     last_ai_message = ai_messages[-1].content if ai_messages else None

#     print(last_ai_message)
    # print(f"Agent: {response["messages"]}")