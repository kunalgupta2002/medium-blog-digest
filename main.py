import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

# Initialize models
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")


# Create Streamlit app
def main():
    st.title("medium blog digest")

    # Input from user
    query = st.text_input("Enter your query:", "What is Vector Database in machine learning?")

    if st.button("Submit"):
        # Use PromptTemplate to create a chain
        prompt_template = PromptTemplate.from_template(template=query)
        chain = prompt_template | llm

        # Invoke the chain
        result = chain.invoke(input={})
        st.write("Result:", result.content)

    # Display vector store details
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"], embedding=embeddings
    )
    st.write("Vector Store Initialized with Index:", os.environ["INDEX_NAME"])


if __name__ == "__main__":
    main()



