"""Apollo is a tool to help you many tasks with AI support"""
import typer

from apollo.tools.document_qa import DocumentQA

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


if __name__ == "__main__":
    app()
