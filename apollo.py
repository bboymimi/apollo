"""Apollo is a tool to help you many tasks with AI support"""
import typer

from apollo.tools.document_qa import DocumentQA
from apollo.tools.rephrase_article import RephraseArticle

app = typer.Typer()


@app.command()
def doc(action: str, file_name: str, llm: str = "openai",
        persist_directory: str = "./chroma"):
    """Load the document from the file"""
    d_qa = DocumentQA()
    if action == "update":
        d_qa.update(file_name, llm, persist_directory)
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


if __name__ == "__main__":
    app()
