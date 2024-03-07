from utils.chat_history_manager import MessageManager, Message, UserMessage, AIMessage, SystemMessage
from utils.tools import ChineseLawDBTool
from utils.chatbot import Chatbot
import config

# tools
tools = [ChineseLawDBTool(path=config.DB_PATH)]
tools_description = [tool.description for tool in tools]
tools_map = {tools_description[i]["function"]["name"]:tools[i] for i in range(len(tools))}

# chat history manager
msg_mgr = MessageManager(max_tokens=config.MAX_TOKENS)
msg_mgr.append(SystemMessage(config.SYSTEM_PROMPT))
for few_shot in config.FEW_SHOT: msg_mgr.append(Message(*few_shot))

# chatbot
chatbot = Chatbot(
    azure_deployment=config.AOAI_DEPLOYMENT,
    api_version = config.AOAI_API_VERSION,
    api_key = config.AOAI_API_KEY, 
    azure_endpoint = config.AOAI_ENDPOINT,
    max_tokens = 1000
)

try:
    while True:
        user_input = input("user>>>")
        msg_mgr.append(UserMessage(user_input))
        ai_output = chatbot(msg_mgr.to_list(), tools_description, tools_map).choices[0].message.content
        print("AI: ", ai_output)
        msg_mgr.append(SystemMessage(ai_output))
except KeyboardInterrupt:
    print("break by user")