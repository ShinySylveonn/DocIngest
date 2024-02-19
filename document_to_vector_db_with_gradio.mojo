from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from gradio import Input, outputs
import pymongo

def load_and_split_document(document_path) {
    loader = TextLoader(document_path)
    pages = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(pages)
    return texts
}

def create_vector_store(texts, embedding) {
    vector_store = Weaviate.from_texts(
        texts=texts,
        embedding=embedding,
        client_options={"url": "http://localhost:8080"},
    )
    return vector_store
}

def store_vector_store_in_db(vector_store, db_connection) {
    db_connection.insert_one({"vector_store": vector_store.as_dict()})
}

def load_vector_store_from_db(db_connection) {
    vector_store_dict = db_connection.find_one({})["vector_store"]
    return Weaviate.from_dict(vector_store_dict)
}

def remove_document_from_vector_store(vector_store, document_id) {
    vector_store.delete_by_id(document_id)
}

def store_document_in_vector_db(document_path, embedding, db_connection) {
    texts = load_and_split_document(document_path)
    vector_store = create_vector_store(texts, embedding)
    store_vector_store_in_db(vector_store, db_connection)
    return vector_store
}

def create_retrieval_qa_chain(db_connection) {
    vector_store = load_vector_store_from_db(db_connection)
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=retriever
    )
    return qa_chain
}

def connect_to_db():
    # Replace the connection string with the appropriate connection string for your database
    connection_string = "mongodb://localhost:27017/"
    db_connection = pymongo.MongoClient(connection_string)["document_db"]
    return db_connection

def gradio_interface():
    db_connection = connect_to_db()
    document_path = Input(type="filepath", description="Upload a document")
    query = Input(type="text", description="Enter a query")
    document_id = Input(type="text", description="Enter the document ID to remove")
    output = outputs.Textbox(label="Answer")

    def update_output(document_path, query, document_id, db_connection) {
        if document_id:
            vector_store = load_vector_store_from_db(db_connection)
            remove_document_from_vector_store(vector_store, document_id)
            db_connection.update_one(
                {"_id": db_connection.find_one({})["_id"]},
                {"$set": {"vector_store": vector_store.as_dict()}},
            )
            output.update("Document removed successfully.")
        else:
            vector_store = store_document_in_vector_db(document_path, embedding, db_connection)
            qa_chain = create_retrieval_qa_chain(db_connection)
            answer = qa_chain.run(query)
            output.update(f"Answer: {answer}")

    return gradio_interface(
        fn=update_output,
        inputs=[document_path, query, document_id, db_connection],
        outputs=output,
    )

gradio_interface().launch()
