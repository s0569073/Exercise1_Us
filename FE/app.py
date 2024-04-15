# https://docs.chainlit.io/get-started/pure-python

import chainlit as cl
from transformers import pipeline

generator = pipeline(task="text-generation")
generator_bloomz = pipeline(model="bigscience/bloomz-1b7")

@cl.on_message
async def main(message: cl.Message):

    # Your custom logic goes here...

    output = generator_bloomz(message.content, return_full_text=False)
    generated_text = output[0]['generated_text']

    # Send a response back to the user
    await cl.Message(
        content=generated_text,
    ).send()




if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit("app.py")