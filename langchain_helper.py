from langchain.llms import GooglePalm
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import SemanticSimilarityExampleSelector
from langchain.chains.sql_database.prompt import PROMPT_SUFFIX,_mysql_prompt,MYSQL_PROMPT
from langchain.prompts.prompt import PromptTemplate
from langchain.prompts import FewShotPromptTemplate
import os
from dotenv import load_dotenv
load_dotenv()
from few_shots import few_shots


def few_shot_chain():
    db_user = "root"
    db_pw= ""
    db_host = "localhost"
    db_port = "3306"
    db_name = "tshirts"

    db = SQLDatabase.from_uri(f"mysql+pymysql://{db_user}:{db_pw}@{db_host}:{db_port}/{db_name}", sample_rows_in_table_info = 3)

    api_key = os.getenv('GOOGLE_API_KEY')
    llm = GooglePalm(google_api_key = api_key, temperature = 0.2)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    e = embeddings.embed_query("how many white t-shirt are there in levis brand?")

    to_vectorize = [" ".join(example.values()) for example in few_shots]

    vectorstore = Chroma.from_texts(to_vectorize, embedding=embeddings, metadatas=few_shots)


    example_selector =SemanticSimilarityExampleSelector(vectorstore = vectorstore, k=2,)

    ex_prompt = PromptTemplate(inputformat = ["Question", "SQLQuery", "SQLResult", "Answer"],
                template = "\nQuestion: {Question}\nSQLQuery: {SQLQuery}\nSQLResult: {SQLResult}\nAnswer: {Answer}")

    fewshot_prompt= FewShotPromptTemplate(
    example_selector = example_selector,example_prompt=ex_prompt,prefix=_mysql_prompt, suffix=PROMPT_SUFFIX, 
        input_variables=['input', 'table_info', 'top_k'],
    )

    new_chain = SQLDatabaseChain.from_llm(llm,db,verbose = True, prompt = fewshot_prompt)

    return new_chain



