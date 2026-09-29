import gradio as gr
from retrieval_pipeline import *
from retrieve_eval import *
import os


def clouddesk_agent(text, number):
    if not text.strip():
        return "", "",""

    
    res = run_rag_pipeline(cfg, vstore, client, text )
    docs = retrieve_docs(vstore, text, k=number)
    context_str, citation_str = format_docs(docs)
    confidence = res['confidence_pct']
    escalated = res['requires_escalation']
    answer = res['answer']
    Source = citation_str
    confidence_output = (
        f"Confidence: {confidence}\n"
        f"Escalation required: {escalated}")
    return answer, confidence_output, Source


cloud_theme = gr.themes.Base(
    primary_hue="pink",
    secondary_hue="cyan",
    neutral_hue="slate" )

with gr.Blocks(
) as demo:

    gr.Image("logo.PNG", width=350, show_label=False,
                  interactive=False,buttons=[])

    gr.HTML(
        """
        <div style="
        background: linear-gradient(135deg,#24245c,#4b4bb8);
        padding: 10px;
        border-radius: 10px;
        text-align:center;
        color:white;
        margin-bottom:10px;
        ">
        <h1 style="
        margin-bottom:8px;
        font-size:32px;
        ">
        ☁ CloudDesk AI Support Engineer
       </h1>
       <p style="
        font-size:16px;
        opacity:0.9;
       ">
        Customer Support Retrieval-Augmented Generation (RAG)
        assistant
        </p>
        </div>
        """
        )
    
    with gr.Row():

        with gr.Column(scale=1):

            gr.HTML("""
            <div style="
          background:#e7e7ff;
          color:#4b4bb8;
          padding:4px 10px;
          border-radius:8px;
          font-weight:bold;
           display:inline-block;
           margin-bottom:2px;
         ">
         Enter Query...  
         </div>
         """)

            query = gr.Textbox(
                label= "Customer Inquiry",
                show_label=False,
            placeholder="e.g. How do I integrate Slack with CloudDesk?",
            )

            gr.HTML("""
                         <div style="
                       background:#e7e7ff;
                       color:#4b4bb8;
                       padding:4px 10px;
                       border-radius:8px;
                       font-weight:bold;
                        display:inline-block;
                        margin-bottom:2px;
                      ">
                      Top-K Source Evidence...  
                      </div>
                      """)
            number = gr.Slider(
                    label= "Top-K Source Evidence",
                    show_label=False,
                    minimum=1,
                    maximum=5,
                    step=1,
                    value=3
                )
        
            submit = gr.Button(
            " 🚀 Submit Query",
            variant="primary"
            )

            clear = gr.Button(
            "🧹 Clear"
            )

            gr.Examples(
             examples=[
                ["My SAML login stopped working after adding a new domain", 1],
                ["How do I integrate Slack with CloudDesk?", 2],
                ["We connected CloudDesk to Slack but new tickets aren't showing", 3],
                ["Why is my API failing with a 401 Unauthorized error?", 4],
                ["How do I manage API keys and rate limits for my team?", 5],
              ],
              inputs=[query, number],
              label="Try an example query.."
              )


        with gr.Column(scale=1):
                   
                answer = gr.Markdown(
                value="Awaiting query..."
                )

                gr.HTML("""
                              <div style="
                              background:#e7e7ff;
                              color:#4b4bb8;
                              padding:4px 10px;
                              border-radius:8px;
                              font-weight:bold;
                              display:inline-block;
                       ">
                             Confidence Level
                        </div>""")
                confidence = gr.Textbox(show_label=False
            )

                gr.HTML("""
                              <div style="
                              background:#e7e7ff;
                              color:#4b4bb8;
                              padding:4px 10px;
                              border-radius:8px;
                              font-weight:bold;
                              display:inline-block;
                       ">
                             📌 Source Evidence
                        </div>""")
                source = gr.Markdown(
                label="📌 Source Evidence" 
                #show_label=False,
                  )


    submit.click(
    fn=clouddesk_agent,
    inputs=[
        query,
        number
    ],
    outputs=[
        answer,
        confidence,
        source
    ]
     )

    clear.click(
    fn=lambda: ("", "Awaiting query...", "","", 3),
    inputs=[],
    outputs=[
        query,
        answer,
        confidence,
        source,
        number
    ]
    )

PORT = int(os.environ.get("PORT", 7860))

demo.launch(share=True,
            allowed_paths=["./"],
            theme=cloud_theme,
            server_name="0.0.0.0",
            server_port=PORT) 
