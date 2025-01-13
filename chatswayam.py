import os
import logging
from langchain_community.document_loaders import UnstructuredCSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever


# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = "/Users/rushik/csv rag/data/FullInventory.csv"
MODEL_NAME = "gpt-3.5-turbo"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"


def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredCSVLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        return None


def split_documents(documents):
    """Split documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks


def create_vector_db(chunks):
    """Create a vector database from document chunks."""
    # Pull the embedding model if not already available
    ollama.pull(EMBEDDING_MODEL)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=EMBEDDING_MODEL),
        collection_name=VECTOR_STORE_NAME,
    )
    logging.info("Vector database created.")
    return vector_db


def create_retriever(vector_db, llm):
    """Create a multi-query retriever."""
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template=""" Your task is to identify products 
        and product id's from given list,
        you have construction industry experience from Gujarat,
        you can identify items in local names also.

Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever


def create_chain(retriever, llm):
    """Create the chain"""
    # RAG prompt
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created successfully.")
    return chain


def main():
    # Load and process the PDF document
    data = ingest_pdf(DOC_PATH)
    if data is None:
        return

    # Split the documents into chunks
    chunks = split_documents(data)

    # Create the vector database
    vector_db = create_vector_db(chunks)

    # Initialize the language model
    llm = ChatOllama(model=MODEL_NAME)

    # Create the retriever
    retriever = create_retriever(vector_db, llm)

    # Create the chain with preserved syntax
    chain = create_chain(retriever, llm)

    # Example query
    question = """ Task:

I will provide you with a list of items in inventory and a raw customer order. Your task is to generate a JSON output and not give me ANY other explanation where:

Rules:
1. Specific Items:
Match the customer order to inventory items exactly by name, size, or type.
Include only the matching VUID, name, and specified qty.Ensure partial matches are cross-referenced against the inventory for existing products before marking as unmatched.
2. Ambiguous Items:
If the order is vague (e.g., "Shoes" without a specified size), include all matching inventory items.Always check for partial matches in the inventory list before marking an item as unmatched.
3. Non-Matching Items:
If an item does not match any inventory item, use nil as the VUID.Cross-reference all potential matches with inventory descriptions to avoid overlooking existing matches.
Set the name from the customer's description and include the specified qty.
4. Ambiguous Sizes or Types:
If an item request includes a type (e.g., "Shoes") but does not specify size, match all relevant inventory items with that type.
5.Handling Ambiguous Quantities:
If the quantity in the customer order is ambiguous (e.g., "5 box"):
Extract only the numeric portion of the quantity (e.g., "5" from "5 box").
Use the numeric value as qty in the output.
Output Format:
The output must be in JSON format with the following structure:
{
    "<product nane used by customer>": {
        "qty": <quantity requested by the customer>,
        "matching_products": [{
        "VUID": "<VUID from our list of products>",
"name: "<Name from our list of products>"
        }]
    },
    ...
}


Customer Order:

Ronark buildwell 

Site- alankar (sanand)

Safety Shoes x 7
Pawda no handle x 5
Damar x 5 box
Ply Wood 30x50m x 5 """

    # Get the response
    res = chain.invoke(input=question)
    print("Response:")
    print(res)


if __name__ == "__main__":
    main()