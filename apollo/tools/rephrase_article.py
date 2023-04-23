"""This script demonstrates how to to rephrase a document."""

from langchain import OpenAI
from langchain.chains.llm import LLMChain
from langchain.document_loaders import UnstructuredFileLoader
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

from keys import OPENAI_API_KEY

BUSINESS_TEMPLATE = ("Rephrase the following article by a polite and"
                     " professional tone. Please remember to correct the"
                     " grammatical errors, use advanced grammatical clauses, and"
                     " make it expressive and detailed. Also, please improve the"
                     " wording to make it look like highly-educated native"
                     " speakers'"
                     " essay. It is very important that the quality"
                     " need to be very rigourous and the legnth needs to be 1.5"
                     " times of the origianl number of words. For example, the"
                     " origianl legnth is 100 words, and the rephrased version"
                     " needs to be at least 150 words. Also, it's wonderful if"
                     " you can separate into several paragraphs for better"
                     " understanding. This is ready for Elon Musk to review."
                     " This should be in the following format:\n"
                     " Original article:[original content]\n\n"
                     " Rephrased result:[rephrased content]\n\n"
                     " Example:\n"
                     " Original article:The candidate has some KVM/Qemu/IOMMU/VIRTIO experiences and seems ambitious about the developer-type work. The candidate shows dedication, focus, persistency, and technical depth from the challenging bug resolution examples. Also, in the APAC region, we still need more people familiar with the virtualization realm. Based on the description, I'll rank the candidate between Software Engineer or Software Engineer II. In the hiring process, we need to confirm that the candidate would like to work on the interrupt-driven sustaining engineering type of work (as mentioned in the written interview, the candidate would like to focus on the virtualization field and thought Canonical would have free time to do the community work). In the candidate's experience, only development type of work was demonstrated and didn't have any customer-facing and pure bug-fixing oriented work - Some developers would feel this is the support engineer instead of a software developer.\n\n"
                     " Rephrased result:The candidate possesses a wealth of experience in KVM/Qemu/IOMMU/VIRTIO and demonstrates a keen ambition to engage in developer-centric work. Their commitment, concentration, persistence, and technical acumen are evident from the complex bug resolution examples provided. Furthermore, there is a current need for individuals with expertise in virtualization within the APAC region. Upon evaluating the candidate's background and skills, it appears appropriate to consider them for a position as either a Software Engineer or Software Engineer II. However, during the hiring process, it is crucial to ascertain whether the candidate is genuinely interested in engaging with interrupt-driven sustaining engineering tasks. While the written interview suggests a desire to concentrate on the virtualization domain and an assumption that Canonical would provide ample opportunity for community work, it is essential to verify the candidate's expectations. It is worth noting that the candidate's experience primarily pertains to development work, with no explicit mention of customer-facing or bug-fixing focused tasks. Consequently, some developers might perceive the role in question to be more akin to that of a support engineer rather than a software developer. Ensuring alignment between the candidate's preferences and the position's responsibilities is of utmost importance in this hiring process.\n\n"
                     " Begin!\n"
                     " Original article: {content}\n\n"
                     " Rephrased result:\n")

CASUAL_TEMPLATE = ("Rephrase the following article by a casual and"
                   " friendly tone. Please remember to correct the"
                   " grammatical errors, use advanced grammatical clauses, and"
                   " make it expressive and detailed. Also, please improve the"
                   " wording to make it look like highly-educated native"
                   " speakers'"
                   " essay. It is very important that the quality"
                   " need to be very rigourous and the legnth needs to be 1.5"
                   " times of the origianl number of words. For example, the"
                   " origianl legnth is 100 words, and the rephrased version"
                   " needs to be at least 150 words. Also, it's wonderful if"
                   " you can separate into several paragraphs for better"
                   " understanding. This is ready for Elon Musk to review."
                   " This should be in the following format:\n"
                   " Original article:[original content]\n\n"
                   " Rephrased result:[rephrased content]\n\n"
                   " Begin!\n"
                   " Original article: {content}\n\n"
                   " Rephrased result:\n")


class RephraseArticle:
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

        # print out words count and token count (num_words * (4/3))
        print(f"You have roughly {num_words} words or {num_words * (4 / 3)}"
              f" tokens in your docs")
        print()

    def run(self, file_name: str, tone: str = "business", debug: bool = False):
        """Remove the path from the name also the subfix with .*"""

        llm_model = OpenAI(openai_api_key=OPENAI_API_KEY,
                           temperature=1, max_tokens=-1)
        loader = UnstructuredFileLoader(file_name)
        doc = loader.load()
        content = doc[0].page_content
        self.doc_summary(doc)
        if tone == "business":
            print("Rephrasing the document in a business tone:")
            template = BUSINESS_TEMPLATE
            prompt = PromptTemplate(template=template, input_variables=[
                                    "content"])
        elif tone == "casual":
            print("Rephrasing the document in a casual tone:")
            template = CASUAL_TEMPLATE
            prompt = PromptTemplate(template=template, input_variables=[
                                    "content"])

        llm_chain = LLMChain(prompt=prompt, llm=llm_model, verbose=debug)

        result = llm_chain.predict(content=content)
        doc = [Document(page_content=result, metadata={"source": None})]

        self.doc_summary(doc)
        print(f"\"{result}\"\n")

        # Output the result into a file with designated directory
        with open(f"{file_name}.rephrased", "w", encoding="utf-8") as f:
            f.write(result)
        print(f"The rephrased document has been saved in"
              f" {file_name}.rephrased\n")
