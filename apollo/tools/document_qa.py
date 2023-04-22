"""This script demonstrates how to to answer questions about a document."""

from langchain import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

from keys import OPENAI_API_KEY

QA_PROMPT = """Given the following extracted parts of a long document and a question, create a final answer.
You must provide the answer based on the provided content. Please just say that you don't know if if cannot be found in the provided content. Don't try to make up an answer.
In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

QUESTION: [question here]
Content: [document here]
FINAL ANSWER: [answer here]
SCORE: [score between 0 and 100]

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.

Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.

Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
=========
FINAL ANSWER: This Agreement is governed by English law.
SCORE: 100

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.

Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.

Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.

Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
=========
FINAL ANSWER: The content did not mention Michael Jackson. The content is a speech given by an American leader addressing the country on several issues. The speech begins with acknowledging the COVID-19 pandemic, the loss of lives, and the need to unite as Americans to combat it. The leader then mentions the recent invasion of Ukraine by Russia and praises the courage of the Ukrainian people in defending their homeland. The leader assures Americans that the US is taking robust action to protect American businesses and consumers and also announces the release of 30 million barrels of oil from the Strategic Petroleum Reserve to help blunt gas prices. The speech then calls on Congress to fund ARPA-H to drive breakthroughs in healthcare. The leader concludes the speech by talking about the sacredness of the Capitol and the responsibility of the American people to continue building a strong, prosperous nation.
SCORE: 100


Begin!

QUESTION: {question}
=========
{context}
=========
FINAL ANSWER:"""


class DocumentQA:
    """A class to handle document QA tasks."""

    def doc_summary(self, docs):
        """
        Generate a summary of the given document.

        Args:
            docs (list): A list of strings representing the document.

        Returns:
            None
        """

        print(f"You have {len(docs)} document(s)")

        # this is a list comprehension
        num_words = sum([len(doc.page_content.split(" ")) for doc in docs])

        print(f"You have roughly {num_words} words in your docs")
        print()

    def run(self, name: str, llm: str = "openai",
            persist_directory: str = "./chroma"):
        """Remove the path from the name also the subfix with .*"""
        name = name.split("/")[-1].split(".")[0]

        llm_model = OpenAI(openai_api_key=OPENAI_API_KEY)

        # transfer llm to lowercase and compare
        llm = llm.lower().strip()
        if llm == "openai":
            print("Using OpenAI as the embedding model")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        elif llm == "huggingface":
            print("Using HuggingFace as the embedding model")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-distilroberta-v1"
            )
        else:
            print("Using default OpenAI embedding model")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        docsearch = Chroma(
            collection_name=name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )

        output_parser = RegexParser(
            regex=r"(.*?)\nSCORE: (.*)",
            output_keys=["answer", "score"],
        )
        prompt = PromptTemplate(
            template=QA_PROMPT,
            input_variables=["context", "question"],
            output_parser=output_parser,
        )
        chain = load_qa_chain(
            llm_model,
            chain_type="map_rerank",
            verbose=False,
            return_intermediate_steps=False,
            prompt=prompt,
        )

        while True:
            question = input("Ask> ")

            if question.lower().strip() in ["exit", "quit", "q", "e"]:
                exit(0)

            doc = docsearch.similarity_search(question, k=5)
            doc.append(Document(page_content=question,
                       metadata={"source": None}))
            result = chain(
                {"input_documents": doc, "question": question},
                return_only_outputs=True)
            print(result["output_text"])

    def update(self, name: str, llm: str = "openai",
               persist_directory: str = "./chroma"):
        """Load the document from the file"""
        loader = UnstructuredFileLoader(name)
        doc = loader.load()

        # Show the summary of the document
        self.doc_summary(doc)
        spliter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        text = spliter.split_documents(doc)

        # Remove the path from the name also the subfix with .*
        name = name.split("/")[-1].split(".")[0]

        # transfer llm to lowercase and compare
        llm = llm.lower().strip()
        if llm == "openai":
            print("Using OpenAI as the embedding model")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        elif llm == "huggingface":
            print("Using HuggingFace as the embedding model")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-distilroberta-v1"
            )
        else:
            print("Using default OpenAI embedding model")
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        # By utilizing Chroma with split documents, a vector store transferred
        # with embeddings API is created.
        docsearch = Chroma.from_documents(
            collection_name=name,
            documents=text,
            embedding=embeddings,
            persist_directory=persist_directory,
        )

        # Persist the vector store
        docsearch.persist()
