from openai import AzureOpenAI
import json

class Chatbot:
    def __init__(self, azure_deployment, max_tokens, **kwargs) -> None:
        self.client = AzureOpenAI(**kwargs)
        self.max_tokens = max_tokens
        self.azure_deployment = azure_deployment

    def __call__(self, messages, tools_description:list, tools:dict):
        response = self.client.chat.completions.create(
            model=self.azure_deployment,
            messages=messages,
            tools=tools_description,
            tool_choice="auto",  # auto is default, but we'll be explicit
            temperature = 0.1,
            max_tokens=self.max_tokens
        )

        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        # Step 2: check if the model wanted to call a function
        print(tool_calls)
        if tool_calls:
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            messages.append(response_message)  # extend conversation with assistant's reply
            # Step 4: send the info for each function call and function response to the model
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_to_call = tools[function_name]
                function_args = json.loads(tool_call.function.arguments)
                function_response = function_to_call(
                    query=function_args.get("query"),
                )
                messages.append(
                    {
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": function_name,
                        "content": function_response,
                    }
                )  # extend conversation with function response
            second_response = self.client.chat.completions.create(
                model=self.azure_deployment,
                messages=messages,
            )  # get a new response from the model where it can see the function response
            return second_response
        else:
            return response