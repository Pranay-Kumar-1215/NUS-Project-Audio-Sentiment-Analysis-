import gradio as gr

import random


def greet(name):
    myList = ["This is positive","This is negative","This is neutral"]
    random_element = random.choice(myList)
    return random_element

demo = gr.Interface(fn=greet, inputs="text", outputs="text")

demo.launch()
