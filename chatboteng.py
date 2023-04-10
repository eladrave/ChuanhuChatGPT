import os
import logging
import sys

import gradio as gr

from modules import config
from modules.config import *
from modules.utils import *
from modules.presets import *
from modules.overwrites import *
from modules.chat_func import *
from modules.openai_func import get_usage

gr.Chatbot.postprocess = postprocess
PromptHelper.compact_text_chunks = compact_text_chunks

with open("assets/custom.css", "r", encoding="utf-8") as f:
    customCSS = f.read()

with gr.Blocks(css=customCSS, theme=small_and_beautiful_theme) as demo:
    user_name = gr.State("")
    history = gr.State([])
    token_count = gr.State([])
    promptTemplates = gr.State(load_template(
    get_template_names(plain=True)[0], mode=2))
    # Create a variable to store the API key
    user_api_key = gr.State(my_api_key)
    # Create a variable to store the user's question
    user_question = gr.State("")
    # Create a variable to store whether the user is currently outputing the conversation history
    outputing = gr.State(False)
    # Create a variable to store the user's conversation history's topic
    topic = gr.State("Unnamed Conversation History")
    with gr.Row():
        gr.HTML(title, elem_id="app_title")
        status_display = gr.Markdown(get_geoip(), elem_id="status_display")
    with gr.Row(elem_id="float_display"):
        user_info = gr.Markdown(value="getting user info...", elem_id="user_info")

        # https://github.com/gradio-app/gradio/pull/3296
        def create_greeting(request: gr.Request):
            if hasattr(request, "username") and request.username:  # is not None or is not ""
                logging.info(f"Get User Name: {request.username}")
                return gr.Markdown.update(value=f"User: {request.username}"), request.username
            else:
                return gr.Markdown.update(value=f"User: default", visible=False), ""
        demo.load(create_greeting, inputs=None, outputs=[user_info, user_name])

    with gr.Row().style(equal_height=True):
        with gr.Column(scale=5):
            with gr.Row():
                chatbot = gr.Chatbot(
                    elem_id="chuanhu_chatbot").style(height="100%")
            with gr.Row():
                with gr.Column(min_width=225, scale=12):
                    user_input = gr.Textbox(
                        elem_id="user_input_tb",
                        show_label=False, placeholder="Enter here"
                    ).style(container=False)
                with gr.Column(min_width=42, scale=1):
                    submitBtn = gr.Button(
                        value="", variant="primary", elem_id="submit_btn")
                    cancelBtn = gr.Button(
                        value="", variant="secondary", visible=False, elem_id="cancel_btn")
            with gr.Row():
                emptyBtn = gr.Button(
                    "🧹 New Conversation",
                )
                retryBtn = gr.Button("🔄 Regenerate")
                delFirstBtn = gr.Button("🗑️ Delete Oldest Conversation")
                delLastBtn = gr.Button("🗑️ Delete Newest Conversation")
                reduceTokenBtn = gr.Button("♻️ Summarize Conversation")

        with gr.Column():
            with gr.Column(min_width=50, scale=1):
                with gr.Tab(label="ChatGPT"):
                    keyTxt = gr.Textbox(
                        show_label=True,
                        placeholder=f"OpenAI API-key...",
                        value=hide_middle_chars(my_api_key),
                        type="password",
                        visible=not HIDE_MY_KEY,
                        label="API-Key",
                    )
                    if multi_api_key:
                        usageTxt = gr.Markdown(
                            "Multi-account mode enabled, no need to input key, you can start a conversation directly", elem_id="usage_display")
                    else:
                        usageTxt = gr.Markdown(
                            "**Send message** or **Submit key** to display quota", elem_id="usage_display")
                    model_select_dropdown = gr.Dropdown(
                        label="Select Model", choices=MODELS, multiselect=False, value=MODELS[0]
                    )
                    use_streaming_checkbox = gr.Checkbox(
                        label="Real-time transmission of answers", value=True, visible=enable_streaming_option
                    )
                    use_websearch_checkbox = gr.Checkbox(
                        label="Use online search", value=False)
                    language_select_dropdown = gr.Dropdown(
                        label="Select reply language (for search & index functions)",
                        choices=REPLY_LANGUAGES,
                        multiselect=False,
                        value=REPLY_LANGUAGES[0],
                    )
                    index_files = gr.Files(
                        label="Upload index files", type="file", multiple=True)
                    two_column = gr.Checkbox(
                        label="Double-column pdf", value=advance_docs["pdf"].get("two_column", False))
                    # TODO: Formula OCR
                    # formula_ocr = gr.Checkbox(label="Recognize formula", value=advance_docs["pdf"].get("formula_ocr", False))

                with gr.Tab(label="Prompt"):
                    systemPromptTxt = gr.Textbox(
                        show_label=True,
                        placeholder=f"Enter System Prompt here...",
                        label="System prompt",
                        value=initial_prompt,
                        lines=10,
                    ).style(container=False)
                    with gr.Accordion(label="Load Prompt Template", open=True):
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=6):
                                    templateFileSelectDropdown = gr.Dropdown(
                                        label="Select Prompt template collection file",
                                        choices=get_template_names(plain=True),
                                        multiselect=False,
                                        value=get_template_names(plain=True)[0],
                                    ).style(container=False)
                                with gr.Column(scale=1):
                                    templateRefreshBtn = gr.Button("🔄 Refresh")
                            with gr.Row():
                                with gr.Column():
                                    templateSelectDropdown = gr.Dropdown(
                                        label="Load from Prompt template",
                                        choices=load_template(
                                            get_template_names(plain=True)[
                                                0], mode=1
                                        ),
                                        multiselect=False,
                                    ).style(container=False)

                with gr.Tab(label="Save/Load"):
                    with gr.Accordion(label="Save/Load conversation history", open=True):
                        with gr.Column():
                            with gr.Row():
                                with gr.Column(scale=6):
                                    historyFileSelectDropdown = gr.Dropdown(
                                        label="Load conversation from list",
                                        choices=get_history_names(plain=True),
                                        multiselect=False,
                                        value=get_history_names(plain=True)[0],
                                    )
                                with gr.Column(scale=1):
                                    historyRefreshBtn = gr.Button("🔄 Refresh")
                            with gr.Row():
                                with gr.Column(scale=6):
                                    saveFileName = gr.Textbox(
                                        show_label=True,
                                        placeholder=f"Set file name: Default is .json, optional is .md",
                                        label="Set save file name",
                                        value="Conversation history",
                                    ).style(container=True)
                                with gr.Column(scale=1):
                                    saveHistoryBtn = gr.Button(
                                        "💾 Save conversation")
                                    exportMarkdownBtn = gr.Button(
                                        "📝 Export as Markdown")
                                    gr.Markdown(
                                        "Saved by default in the history folder")
                            with gr.Row():
                                with gr.Column():
                                    downloadFile = gr.File(interactive=True)

                with gr.Tab(label="Advanced"):
                    gr.Markdown(
                        "# ⚠️ Proceed with caution ⚠️\n\nIf not working, please restore default settings")
                    default_btn = gr.Button("🔙 Restore default settings")

                    with gr.Accordion("Parameters", open=False):
                        top_p = gr.Slider(
                            minimum=-0,
                            maximum=1.0,
                            value=1.0,
                            step=0.05,
                            interactive=True,
                            label="Top-p",
                        )
                        temperature = gr.Slider(
                            minimum=-0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                            interactive=True,
                            label="Temperature",
                        )

                    with gr.Accordion("Network settings", open=False):
                        # Prioritize displaying custom api_host
                        apihostTxt = gr.Textbox(
                            show_label=True,
                            placeholder=f"Enter API-Host here...",
                            label="API-Host",
                            value=config.api_host or shared.API_HOST,
                            lines=1,
                        )
                        changeAPIURLBtn = gr.Button("🔄 Switch API address")
                        proxyTxt = gr.Textbox(
                            show_label=True,
                            placeholder=f"Enter proxy address here...",
                            label="Proxy address (example: http://127.0.0.1:10809)",
                            value="",
                            lines=2,
                        )
                        changeProxyBtn = gr.Button("🔄 Set proxy address")

    gr.Markdown(description)
    gr.HTML(footer.format(versions=versions_html()), elem_id="footer")
    chatgpt_predict_args = dict(
        fn=predict,
        inputs=[
            user_api_key,
            systemPromptTxt,
            history,
            user_question,
            chatbot,
            token_count,
            top_p,
            temperature,
            use_streaming_checkbox,
            model_select_dropdown,
            use_websearch_checkbox,
            index_files,
            language_select_dropdown,
        ],
        outputs=[chatbot, history, status_display, token_count],
        show_progress=True,
    )

    start_outputing_args = dict(
        fn=start_outputing,
        inputs=[],
        outputs=[submitBtn, cancelBtn],
        show_progress=True,
    )

    end_outputing_args = dict(
        fn=end_outputing, inputs=[], outputs=[submitBtn, cancelBtn]
    )

    reset_textbox_args = dict(
        fn=reset_textbox, inputs=[], outputs=[user_input]
    )

    transfer_input_args = dict(
        fn=transfer_input, inputs=[user_input], outputs=[
            user_question, user_input, submitBtn, cancelBtn], show_progress=True
    )

    get_usage_args = dict(
        fn=get_usage, inputs=[user_api_key], outputs=[
            usageTxt], show_progress=False
    )

    # Chatbot
    cancelBtn.click(cancel_outputing, [], [])

    user_input.submit(**transfer_input_args).then(**
                                                chatgpt_predict_args).then(**end_outputing_args)
    user_input.submit(**get_usage_args)

    submitBtn.click(**transfer_input_args).then(**
                                                chatgpt_predict_args).then(**end_outputing_args)
    submitBtn.click(**get_usage_args)

    emptyBtn.click(
        reset_state,
        outputs=[chatbot, history, token_count, status_display],
        show_progress=True,
    )
    emptyBtn.click(**reset_textbox_args)

    retryBtn.click(**start_outputing_args).then(
        retry,
        [
            user_api_key,
            systemPromptTxt,
            history,
            chatbot,
            token_count,
            top_p,
            temperature,
            use_streaming_checkbox,
            model_select_dropdown,
            language_select_dropdown,
        ],
        [chatbot, history, status_display, token_count],
        show_progress=True,
    ).then(**end_outputing_args)
    retryBtn.click(**get_usage_args)

    delFirstBtn.click(
        delete_first_conversation,
        [history, token_count],
        [history, token_count, status_display],
    )

    delLastBtn.click(
        delete_last_conversation,
        [chatbot, history, token_count],
        [chatbot, history, token_count, status_display],
        show_progress=True,
    )

    reduceTokenBtn.click(
        reduce_token_size,
        [
            user_api_key,
            systemPromptTxt,
            history,
            chatbot,
            token_count,
            top_p,
            temperature,
            gr.State(sum(token_count.value[-4:])),
            model_select_dropdown,
            language_select_dropdown,
        ],
        [chatbot, history, status_display, token_count],
        show_progress=True,
    )
    reduceTokenBtn.click(**get_usage_args)

    two_column.change(update_doc_config, [two_column], None)

    # ChatGPT
    keyTxt.change(submit_key, keyTxt, [user_api_key, status_display]).then(
        **get_usage_args)
    keyTxt.submit(**get_usage_args)

    # Template
    templateRefreshBtn.click(get_template_names, None, [
        templateFileSelectDropdown])
    templateFileSelectDropdown.change(
        load_template,
        [templateFileSelectDropdown],
        [promptTemplates, templateSelectDropdown],
        show_progress=True,
    )
    templateSelectDropdown.change(
        get_template_content,
        [promptTemplates, templateSelectDropdown, systemPromptTxt],
        [systemPromptTxt],
        show_progress=True,
    )

    # S&L
    saveHistoryBtn.click(
        save_chat_history,
        [saveFileName, systemPromptTxt, history, chatbot, user_name],
        downloadFile,
        show_progress=True,
    )
    saveHistoryBtn.click(get_history_names, [gr.State(
        False), user_name], [historyFileSelectDropdown])
    exportMarkdownBtn.click(
        export_markdown,
        [saveFileName, systemPromptTxt, history, chatbot, user_name],
        downloadFile,
        show_progress=True,
    )
    historyRefreshBtn.click(get_history_names, [gr.State(
        False), user_name], [historyFileSelectDropdown])
    historyFileSelectDropdown.change(
        load_chat_history,
        [historyFileSelectDropdown, systemPromptTxt, history, chatbot, user_name],
        [saveFileName, systemPromptTxt, history, chatbot],
        show_progress=True,
    )
    downloadFile.change(
        load_chat_history,
        [downloadFile, systemPromptTxt, history, chatbot, user_name],
        [saveFileName, systemPromptTxt, history, chatbot],
    )

    # Advanced
    default_btn.click(
        reset_default, [], [apihostTxt, proxyTxt, status_display], show_progress=True
    )
    changeAPIURLBtn.click(
        change_api_host,
        [apihostTxt],
        [status_display],
        show_progress=True,
    )
    changeProxyBtn.click(
        change_proxy,
        [proxyTxt],
        [status_display],
        show_progress=True,
    )

    logging.info(
        colorama.Back.GREEN
        + "\nChuanhu's friendly reminder: visit http://localhost:7860 to view the interface"
        + colorama.Style.RESET_ALL
    )
    # By default, start the local server, can be accessed directly from IP, do not create public share link
    demo.title = "Chuanhu ChatGPT 🚀"

if __name__ == "__main__":
    reload_javascript()
    # if running in Docker
    if dockerflag:
        if authflag:
            demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
                server_name="0.0.0.0",
                server_port=7860,
                auth=auth_list,
                favicon_path="./assets/favicon.ico",
            )
        else:
            demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
                server_name="0.0.0.0",
                server_port=7860,
                share=False,
                favicon_path="./assets/favicon.ico",
            )
    # if not running in Docker
    else:
        if authflag:
            demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
                share=False,
                auth=auth_list,
                favicon_path="./assets/favicon.ico",
                inbrowser=True,
            )
        else:
            demo.queue(concurrency_count=CONCURRENT_COUNT).launch(
                share=False, favicon_path="./assets/favicon.ico", inbrowser=True
            )  # Change share=True to create a public share link
        # demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0", server_port=7860, share=False) # Custom port can be set
        # demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0
