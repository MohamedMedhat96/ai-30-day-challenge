**Day 0:**: Initial setup and planning done. Created project repository and outlined the main objectives for the project. Set up the development environment and installed necessary dependencies.

**Day 1:**: Today I learned that the API is stateless and a "conversation" is something I reconstruct every call.

**Day 2:**:: Learned about the structure of messages being sent to claude and how to control the messages. I have learned about stop sequence, temperature and top p. Temperature controls the randomness and how likely claude will take the highest probable token next. Using temperature 0 should be used for extraction, code generation etc and higher temperatures should be used for brain storming and creative tasks. I have also learned more about system and user roles, where system defines how the LLM and act and user basically describes the user input

**Day 3:**:  Learned about the four elements that I should use to ensure a better quality output from the AI 1. Explicit instructions 2. Provide example of input and output 3. Use delimiters (xml tags) to differentiate between system requirements and examples and input 4. let it reason by asking it to think step by step 

**Day 4**: Ensure that the format is strictly specified in the prompt and always validate that the format is correct. For example ask for explicit json without markdown
**Day 5**: Streaming the claude response and token account.
**Day 6**: How to save tokens and improve performance? 1. Context management compact history, rules and avoid writing long system prompts so the agent wouldn't lose the context in the middle. 2. Caching system prompts and reusable rules to utilize cache_read_input_tokens which are cheaper than regular input token. Haiku requires a large context window to be cached (5000)