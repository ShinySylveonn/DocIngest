from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from gradio import Input, outputs
import pymongo

def load_and_split_document(document_path):
    loader = TextLoader(document_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    return texts

def create_vector_store(texts, embedding):
    vector_store = Weaviate.from_texts(
        texts=texts,
        embedding=embedding,
        client_options={"url": "http://localhost:8080"},
    )
    return vector_store

def store_document_in_vector_db(document_path, embedding):
    texts = load_and_split_document(document_path)
    vector_store = create_vector_store(texts, embedding)
    return vector_store

embedding = OpenAIEmbeddings()

def create_retrieval_qa_chain(vector_store):
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever
    )
    return qa_chain

def gradio_interface():
    document_path = Input(type="filepath", description="Upload a document")
    query = Input(type="text", description="Enter a query")
    output = outputs.Textbox(label="Answer")

    def update_output(document_path, query):
        vector_store = store_document_in_vector_db(document_path, embedding)
        qa_chain = create_retrieval_qa_chain(vector_store)
        answer = qa_chain.run(query)
        output.update(f"Answer: {answer}")

    return gradio_interface(fn=update_output, inputs=[document_path, query], outputs=output)

def remove_document_from_vector_db(document_id, vector_store):
    vector_store.delete(documents=[document_id])

def main():
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["vector_db"]
    collection = db["documents"]

    if "vector_db" not in db.list_collection_names():
        db.create_collection("vector_db")

    if "documents" not in db.list_collection_names():
        db.create_collection("documents")

    gradio_interface().launch()

if __name__ == "__main__":
    main()
