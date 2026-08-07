import os
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from dotenv import load_dotenv
load_dotenv()

def load_llm_model():
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    llm_model =  ChatMistralAI(name = "mistral-small-2506", api_key = MISTRAL_API_KEY, temperature = 0.3)
    return llm_model

def split_transcript(transcript : str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size = 4000, chunk_overlap = 500)
    transcript_splitter_chunk = splitter.split_text(transcript)
    return transcript_splitter_chunk

def summarize(transcript : str) -> str:
    llm_model = load_llm_model()

    map_prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "Summarize this portion of a meeting/audio transcript concisely."),
        ("human", "{text}")
        ]
    )

    map_chain = map_prompt | llm_model | StrOutputParser()

    chunks = split_transcript(transcript = transcript)

    chunk_summaries = [map_chain.invoke({"text" : chunk}) for chunk in chunks]

    summary = "\n".join(chunk_summaries)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
        ("system","You are an expert meeting summarizer. Combine these partial summaries into one final professional meeting summary in bullet points."),
        ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text":x}) | combined_prompt | llm_model | StrOutputParser()
    )

    final_summary = combined_chain.invoke(summary)

    return final_summary

def generate_title(transcipt : str) -> str:
    llm_model = load_llm_model()

    title_chain = (
        RunnablePassthrough() | RunnableLambda(lambda x:{"text":x}) | 
        ChatPromptTemplate.from_messages([
             (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
            ),
            ("human", "{text}"),
        ])
        | llm_model
        |StrOutputParser()
    )

    title = title_chain.invoke(transcipt[:2000])
    return title
