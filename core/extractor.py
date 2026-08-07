import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv
load_dotenv()

def load_llm_model():
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    llm_model =  ChatMistralAI(name = "mistral-small-2506", api_key = MISTRAL_API_KEY, temperature = 0.2, max_retries=5)
    return llm_model

def build_chain(system_prompt : str):
    llm_model = load_llm_model()
    prompt = ChatPromptTemplate.from_messages([
        ("system",system_prompt),
        ("human","{text}")
    ])
    chain = (RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) | prompt | llm_model | StrOutputParser())

    return chain

def extract_actionable_items(transcript : str) -> str:
    chain = build_chain(system_prompt = "You are an expert meeting analyst. From the meeting transcript, extract all action items. For each provide:\nTask description\nWho is responsible\nDeadline(if mentioned, else write not specified)\n\nFormat as a numbered list. If none found say 'No action items found'")

    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return chain.invoke(transcript)

def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return chain.invoke(transcript)
