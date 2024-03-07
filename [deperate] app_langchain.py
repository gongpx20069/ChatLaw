from langchain_openai import AzureChatOpenAI
from langchain.memory import ChatMessageHistory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import config

history = ChatMessageHistory()

history.add_message(SystemMessage(content=config.SYSTEM_PROMPT))
print(history.messages)

llm = AzureChatOpenAI(
    api_key=config.AOAI_API_KEY,
    azure_deployment=config.AOAI_DEPLOYMENT,
    azure_endpoint=config.AOAI_ENDPOINT,
    api_version=config.AOAI_API_VERSION
)

try:
    while True:
        user_input = input("user>>>")
        history.add_user_message(user_input)
        ai_output = llm.invoke(history.messages)
        print("AI: ", ai_output.content)
        history.add_ai_message(ai_output)
except KeyboardInterrupt:
    print("break by user")