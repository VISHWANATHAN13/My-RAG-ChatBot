import gradio as gr
from rag import ask_question

def chat(question, history):
    answer, documents = ask_question(question)

    history = history or []

    history.append({
        "role": "user",
        "content": question
    })

    history.append({
        "role": "assistant",
        "content": answer
    })

    source_text = "### Sources\n\n"

    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get(
            "source",
            "Unknown source"
        )

        source_text += f"{i}. `{source}`\n"

    return history, "", source_text


with gr.Blocks() as demo:

    gr.Markdown("# College Knowledge Assistant")

    gr.Markdown(
        "Ask me about college policies, courses, facilities and FAQs."
    )

    chatbot = gr.Chatbot(
        height=500
    )

    sources = gr.Markdown(
        "### Sources\n\n"
    )

    with gr.Row():

        user_input = gr.Textbox(
            placeholder="Message your college assistant...",
            show_label=False,
            scale=7
        )

        submit_btn = gr.Button(
            "Enter",
            variant="primary",
            scale=1
        )

    submit_btn.click(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[
            chatbot,
            user_input,
            sources
        ]
    )

    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot],
        outputs=[
            chatbot,
            user_input,
            sources
        ]
    )


demo.launch()