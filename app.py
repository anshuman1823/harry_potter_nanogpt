import gradio as gr
from sample import sample_function

with gr.Blocks() as interface:
    gr.Markdown("# Harry Potter LLM")
    gr.Markdown("### This is a toy version based on NanoGPT github repo by the legend Andrej Karpathy.")
    gr.Markdown("### It is created by training a tiny GPT with a context size of up to 256 characters, 384 feature channels, and 6-layer Transformer with 6 heads in each layer on the Harry Potter series novels using character level tokenization. For more info, refer the github repo.")
    gr.Markdown("### github_repo link: https://github.com/karpathy/nanoGPT")
    
    with gr.Row():
        prompt = gr.Textbox(label="Enter the intial words to start generation...", value = "Harry and Ron")
        gr.Markdown("### Enter starting prompt...Be creative !")

        temperature_slider = gr.Slider(value = 0.8, minimum=0.5, maximum=1.5, label="Temperature")
        gr.Markdown("### Low temperature generates repetitive content...high temperature generates more creative content (might be non-sensical)")
    
    with gr.Row():
        
        top_k = gr.Number(value=200, minimum=100, maximum = 1000, label="top_k")
        gr.Markdown("### k-value for top-k sampling...works similar to temperature (range is 100 to 1000)")
    
    with gr.Row():
        
        num_samples = gr.Number(value=3, minimum=1, maximum=5, precision=0, label="Number of Samples")
        gr.Markdown("### Number of samples to generate")

    output = gr.Textbox(label="Generated Text")

    generate_btn = gr.Button("Generate")

    generate_btn.click(fn=sample_function, 
                       inputs=[prompt, temperature_slider, top_k, num_samples],
                       outputs=output)

interface.launch()