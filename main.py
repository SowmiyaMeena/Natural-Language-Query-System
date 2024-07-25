import streamlit as st
from langchain_helper import few_shot_chain

st.title("T-Shirts: Database Q&A 👕")

question = st.text_input("Question: ")

if question:
    chain = few_shot_chain()
    response = chain.run(question)

    st.header("Answer")
    st.write(response)