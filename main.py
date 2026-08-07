import os
from dotenv import load_dotenv
from utils.audio_processor import process_audio_from_url
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_actionable_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question

from rich import print

load_dotenv()

def run_pipeline(url : str) -> dict:
    print("Starting Vedio Intelligence Assistant.............")

    chunks = process_audio_from_url(path = url)

    transcript = transcribe_all(chunk_paths = chunks)

    title = generate_title(transcipt = transcript)

    summary = summarize(transcript = transcript)

    action_items = extract_actionable_items(transcript = transcript)
    decision_items = extract_key_decisions(transcript = transcript)
    questions = extract_questions(transcript = transcript)

    rag_chain = build_rag_chain(transcript = transcript)

    return {
        "title" : title,
        "transcript" : transcript,
        "summary" : summary,
        "action_items" : action_items,
        "key_decisions" : decision_items,
        "open_questions" : questions,
        "rag_chain" : rag_chain
        }

if __name__=="__main__":
    url = input("Enter the URL : ").strip()
    result = run_pipeline(url = url)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type ['exit','quit','q'] to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!")
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")