from anthropic import Anthropic

client = Anthropic()

resp = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=300,
    messages=[
        {"role": "user", "content": "In 3 sentences, explain what a token is to a software engineer."}
    ],
)

#### Max Tokens set to 10

# resp = client.messages.create(
#     model="claude-haiku-4-5",
#     max_tokens=10,
#     messages=[
#         {"role": "user", "content": "In 3 sentences, explain what a token is to a software engineer."}
#     ],
# )


# resp = client.messages.create(
#     model= "claude-haiku-4-5",
#     max_tokens=10000,
#     messages=[
#         {"role": "user", "content": "what is the difference"},
#         {"role": "user", "content": "if i send multiple messages like this"},
#         {"role": "user", "content": "I currently sent you multiple messages through the client.messages.create SDK for python and I sent multiple messages in the array, how do you handle those different messages?"}
#     ]
# )

### 

# The response 'content' is a LIST of blocks. For a plain text reply, take the first.
print("REPLY:\n", resp.content[0].text)

# This is where the statelessness and cost become real — inspect the usage:
print("\nUSAGE:", resp.usage)          # input_tokens / output_tokens
print("STOP REASON:", resp.stop_reason)  # why generation ended: 'end_turn', 'max_tokens', etc.