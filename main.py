from anthropic import Anthropic

client = Anthropic()  # reads ANTHROPIC_API_KEY from the environment automatically

msg = client.messages.create(
    model="claude-haiku-4-5",          # cheap + fast for daily iteration
    max_tokens=100,
    messages=[{"role": "user", "content": "Reply with exactly: setup works"}],
)
print(msg.content[0].text)