import streamlit as st
from io import StringIO

from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document

class Web:

    def get_file(self):
        st.set_page_config(page_title="Apollo Demo", page_icon=":robot:", layout="wide")
        st.header("Apollo Demo")

        st.write("")
        f_col, opt_col, buf2 = st.columns([1,0.5,2])
        upload = f_col.file_uploader(label="Upload your text file here")
        option = opt_col.selectbox(
                'LLM model',
                ('openai', 'huggingface'))
        button_col, buf1, buf2 = st.columns([1, 1, 2])
        button_doc = button_col.button("Update to vectorstore")

        #st.write("")
        st.write("")

        if button_doc:
            if upload != None:
                stringio = StringIO(upload.getvalue().decode("utf-8"))
                string_data = stringio.read()
                return upload.name, string_data, option
            else:
                st.write("No document uploaded")
        
        if upload != None:
            return upload.name, None, option
        else:
            return None, None, option

    def get_question(self):
        #q_title = '<p style="font-family:sans-serif; font-size: 20px;">Ask questions for your data</p>'
        #st.markdown(q_title, unsafe_allow_html=True)
        t_col, buf1 = st.columns([1, 1.5])
        question = t_col.text_area(label="", placeholder="Ask questions for your data here")
        button = st.button("Ask")

        st.write("")

        a_title = '<p style="font-family:sans-serif; font-size: 24px;">Answer</p>'
        st.markdown(a_title, unsafe_allow_html=True)
        if button:
            return question
        else:
            return None
