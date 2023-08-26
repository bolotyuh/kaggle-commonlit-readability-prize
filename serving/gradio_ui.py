import gradio as gr


def gradio_block(predictor):
    def predict_fn(text):
        return predictor.predict([text])

    return gr.Interface(
        fn=predict_fn,
        inputs=gr.Textbox(placeholder="Enter a sentence here..."),
        outputs=gr.Textbox(label="Score"),
        examples=[["He used his magic to send Khumo to the future."]],
        title="NLP Inference demo",
    )
