from dataclasses import dataclass, asdict, field
from typing import List
import tiktoken

TOKENS_PER_MESSAGE = 3
TOKENS_PER_NAME = 1
token_encoding = tiktoken.get_encoding('cl100k_base')

@dataclass
class Message:
    role:str
    content:str
    _token_count = TOKENS_PER_MESSAGE

    def __post_init__(self):
        # count token
        self._token_count = TOKENS_PER_MESSAGE + len(token_encoding.encode(self.content))

@dataclass
class MessageManager:
    history: List[Message] = field(default_factory=list)
    max_tokens: int = 4096
    _token_count = 0

    def __post_init__(self):
        for msg in self.history:
            self._token_count += TOKENS_PER_MESSAGE
            self._token_count += len(token_encoding.encode(msg.content))

    def __getitem__(self, items):
        self.history[items]

    def to_list(self):
        return asdict(self)["history"]

    def append(self, msg:Message):
        assert msg._token_count < self.max_tokens, f"the message is too long, more than {self.max_tokens} tokens"
        while msg._token_count + self._token_count > self.max_tokens:
            pop_results = self.pop_msg()
            if len(pop_results) <= 0:
                break
        assert msg._token_count + self._token_count < self.max_tokens, f"The sum of system tokens and current message tokens {msg._token_count} excceed max tokens {self.max_tokens}"
        self.history.append(msg)
        self._token_count += msg._token_count

    def pop_msg(self, role:list=["user", "assistant"], n:int=2):        
        i = 0
        results = []
        while True:
            if n <= 0:
                break
            if i < len(self.history) and self.history[i].role in role:
                pop_item = self.history.pop(i)
                results.append(pop_item)
                self._token_count -= pop_item._token_count
                n -= 1
            else:
                i += 1
        return results

def AIMessage(content:str):
    return Message(role="assistant", content=content)

def UserMessage(content:str):
    return Message(role="user", content=content)

def SystemMessage(content:str):
    return Message(role="system", content=content)

if __name__ == "__main__":
    msg_mgr = MessageManager()
    msg_mgr.append(SystemMessage("hello, how are you?"))
    msg_mgr.append(UserMessage("hello, how are you?"))
    msg_mgr.append(AIMessage("everything is ok."))
    print(msg_mgr.history)
    print(msg_mgr.to_list())
    print(msg_mgr.max_tokens, msg_mgr._token_count)
    result = msg_mgr.pop_msg()
    print(result)
    print(msg_mgr.max_tokens, msg_mgr._token_count)