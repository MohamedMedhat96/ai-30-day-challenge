from anthropic import Anthropic

client = Anthropic()

def classify(prompt_bod: str) -> str:
    r = client.messages.create(
        model= "claude-haiku-4-5",
        max_tokens= 50,
        temperature= 0,
        messages= [{"role": "user", "content": prompt_bod}],
    )

    return r.content[0].text.strip()

ticket = "The invoice PDF won't download, I've tried three browsers."

print("WEAK:", classify(f"Classify this support ticket and provide a possible solution: {ticket}"))

print("----------------------------------------------")
# Strong prompt: explicit + structure + few-shot + constrained output
strong = f"""Classify the support ticket into exactly one category:
BILLING, BUG, FEATURE_REQUEST, or ACCOUNT.
Respond with only the category word and one possible recommendation

<examples>
<ticket>I was charged twice this month</ticket> -> BILLING: The second charge will be refunded within 14 days
<ticket>The export button does nothing</ticket> -> BUG: A ticket with created with the support team
</examples>

<ticket>{ticket}</ticket>"""
print("STRONG:", classify(strong))