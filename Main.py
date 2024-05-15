import gradio as gr

with gr.Blocks(css="""
    body {
        margin: 0;
        padding: 0;
        height: 100vh;
        overflow: hidden;
    }
""") as demo:
    with gr.Tab("Translate"):
        gr.HTML("""<iframe src="https://afrinetwork-translate.hf.space" frameborder="0" width="100%" height="100%"></iframe>""")

    with gr.Tab("Search"):
        gr.HTML("""<iframe src="https://afrinetwork-search.hf.space" frameborder="0" width="100%" height="100%"></iframe>""")

    with gr.Tab("Image Generation"):
        gr.HTML("""<iframe src="https://afrinetwork-image.hf.space" frameborder="0" width="100%" height="100%"></iframe>""")

    with gr.Tab("Image Caption"):
        gr.HTML("""<iframe src="https://afrinetwork-caption.hf.space" frameborder="0" width="100%" height="100%"></iframe>""")

    with gr.Tab("Image Vision"):
        gr.HTML("""<iframe src="https://afrinetwork-caption.hf.space" frameborder="0" width="100%" height="100%"></iframe>""")

    with gr.Tab("Video Subtitle"):
        gr.HTML("""<iframe src="https://afrinetwork-subtitle.hf.space" frameborder="0" width="100%" height="100%"></iframe>""")

demo.launch()
