import json
from anthropic import Anthropic
from pydantic import BaseModel, ValidationError

client = Anthropic()

class Ticket(BaseModel):
    category: str
    urgency: int
    summary: str

prompt = """Extract fields from the ticket as JSON with keys:
category (string), urgency (integer 1-5), summary (string under 12 words).You response should not include any markdowns, it should be a JSON object that will be parsed by pydantic make sure. DO NOT SEND MARKDOWNS.""Ticket: "Production is down, customers can't log in, this started 10 minutes ago." """




raw = client.messages.create(
    model="claude-haiku-4-5", max_tokens=200, temperature=0,
    messages=[{"role": "user", "content": prompt}],
).content[0].text

try:
    ticket = Ticket.model_validate_json(raw)
    print("VALID:", ticket)
except ValidationError as e:
    print("MODEL RETURNED BAD DATA:\n", raw, "\n", e)
