import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import gradio as gr
import torch
from src.models.pretrained import PretrainedVQACaptioner

device = "cuda" if torch.cuda.is_available() else "cpu"
model = PretrainedVQACaptioner(device)


def caption_image(image):
    return model.caption(image)


def answer_question(image, question):
    return model.answer(image, question)


with gr.Blocks() as demo:
    gr.Markdown("# Visual QA & Captioning")
    with gr.Row():
        image_input = gr.Image(type="pil")
        with gr.Column():
            caption_out = gr.Textbox(label="Caption")
            caption_btn = gr.Button("Generate Caption")

            question_in = gr.Textbox(label="Ask a question")
            answer_out = gr.Textbox(label="Answer")
            answer_btn = gr.Button("Get Answer")

    caption_btn.click(caption_image, inputs=image_input, outputs=caption_out)
    answer_btn.click(answer_question, inputs=[image_input, question_in], outputs=answer_out)

if __name__ == "__main__":
    demo.launch(share=True)
