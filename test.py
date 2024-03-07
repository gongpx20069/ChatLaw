from dataclasses import dataclass, asdict


@dataclass
class Message:
    role:str
    content:str

def AIMessage(content:str) -> Message:
    return asdict(Message(role="assistant", content=content))

def UserMessage(content:str) -> Message:
    return asdict(Message(role="user", content=content))

def SystemMessage(content:str) -> Message:
    return asdict(Message(role="system", content=content))

messages = [
    SystemMessage("hello, how are you"),
    UserMessage("Hello"),
    AIMessage("Hi")
]

print(messages)