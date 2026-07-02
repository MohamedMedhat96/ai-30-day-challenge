from anthropic import Anthropic


client = Anthropic()

BIG_CONTEXT = "You are a code reviewer. Here is our 2,000-word style guide: " + ("style rule. " * 2000)

def review(snippet: str):
    return client.messages.create(
        model='claude-haiku-4-5', max_tokens=5600, #Haiku starts caching at 5000~
        system=[
            {
                "type": "text",
                "text": BIG_CONTEXT,
                "cache_control": {"type": "ephemeral"}
            }
        ],
         messages=[{"role": "user", "content": f"Review:\n{snippet}"}],
    )
r1 = review("def add(a,b): return a+b")
print("call 1 usage:", r1.usage)   # note cache *creation* tokens
r2 = review("x=[i for i in range(10)]")
print("call 2 usage:", r2.usage)   # note cache *read* tokens (cheaper) on the cached prefix
