import yaml
config = yaml.safe_load(open("config.yaml"))


# You have to sign up and get an API key: https://platform.openai.com/signup
from openai import OpenAI
client = OpenAI(api_key=config['KEYS']['openai'])


# Pricing: https://openai.com/pricing
# Harry Potter "Chamber of Secrets" is about 100K tokens (GPT 3.5: ~ 5 cents)


# Set limits: https://platform.openai.com/account/billing/limits
# Check usage: https://platform.openai.com/account/usage


# OpenAI API documentation: https://platform.openai.com/docs/introduction/overview

response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with "
                                  "creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(response.choices[0].message.content)


response_2 = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=[
    {"role": "user", "content": "Can you interpret the poem?"}
  ]
)

print(response_2.choices[0].message.content)


response_3 = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  messages=[
      {"role": "assistant", "content": response.choices[0].message.content},
      {"role": "user", "content": "Can you interpret the poem?"}
  ]
)

print(response_3.choices[0].message.content)
