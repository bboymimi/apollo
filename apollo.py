"""Apollo is a tool to help you many tasks with AI support"""
import typer
import streamlit as st

from apollo.tools.document_qa import DocumentQA
from apollo.tools.rephrase_article import RephraseArticle
from apollo.tools.web import Web

app = typer.Typer()


@app.command()
def doc(action: str, file_name: str, llm: str = "openai",
        persist_directory: str = "./chroma"):
    """Load the document from the file"""
    d_qa = DocumentQA()
    if action == "update":
        d_qa.update_file(file_name, llm, persist_directory)
    elif action == "run":
        d_qa.run(file_name, llm, persist_directory)
    else:
        print("Invalid action")


@app.command()
def rephrase(file_name: str, debug: bool = False, tone: str = "business"):
    """Rephrase the article"""

    print(f"Rephrasing the article {file_name}")
    rephrase_article = RephraseArticle()
    rephrase_article.run(file_name, tone, debug)

@app.command()
def web(debug: bool = False, llm: str = "openai", tone: str = "business"):
    """Activate web interface"""

    print("Start streamlit")
    web = Web()
    d_qa = DocumentQA()

    name, text, model = web.get_file()
    if name != None and text != None:
        with st.empty():
            st.write("Updating...")
            d_qa.update_text(name, text, llm=model)
            st.write("Done")

    question = web.get_question()

    if name == None:
        st.write("No document uploaded")

    if name and question:
        with st.empty():
            st.write("Thinking...")
            ans = d_qa.get_answer(name, question)
            st.write(ans)

if __name__ == "__main__":
    app()
