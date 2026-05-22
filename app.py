import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from transformers import pipeline, logging
logging.set_verbosity_error()

import gradio as gr

model = pipeline(
    "summarization",
    model="cnicu/t5-small-booksum",
    framework="pt"
)

def predict(text):
    summary = model(text, max_length=50, min_length=25, do_sample=False)
    return summary[0]['summary_text']

with gr.Blocks() as demo:
    gr.Markdown("## Text Summarization with Hugging Face Transformers")
    input_text = gr.Textbox(label="Input Text", lines=10)
    output_text = gr.Textbox(label="Summary", lines=5)
    summarize_button = gr.Button("Summarize")
    
    summarize_button.click(predict, inputs=input_text, outputs=output_text)

demo.launch()