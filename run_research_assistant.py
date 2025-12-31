# run_research_assistant.py
from research_assistant import ResearchAssistant
import json
import time
'''
def main():
    # If you changed filenames or want to rebuild index, set rebuild_index=True
    assistant = ResearchAssistant(rebuild_index=False)

    # 1) General factual query
    q1 = "What is a convolutional neural network and what is it used for?"
    out1 = assistant.ask(q1, top_k=4)
    print("Q1:", q1)
    print("A1:", out1["answer"])
    print("Sources:", out1["sources"])
    print("="*80)

    time.sleep(0.5)

    # 2) Specific follow-up referencing earlier context
    q2 = "According to the documents, what are two advantages of transformers over RNNs?"
    out2 = assistant.ask(q2, top_k=4)
    print("Q2:", q2)
    print("A2:", out2["answer"])
    print("Sources:", out2["sources"])
    print("="*80)

    time.sleep(0.5)

    # 3) Summarization request
    q3 = "Summarize the main ideas about AI applications in healthcare from the loaded files."
    out3 = assistant.ask(q3, top_k=6)
    print("Q3:", q3)
    print("A3:", out3["answer"])
    print("Sources:", out3["sources"])
    print("="*80)

    # Save outputs to a file for the PDF
    demo = {
        "interaction_1": out1,
        "interaction_2": out2,
        "interaction_3": out3
    }
    with open("demo_outputs.json", "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2, ensure_ascii=False)
    print("Demo outputs saved to demo_outputs.json")

if __name__ == "__main__":
    main()'''

from research_assistant import ResearchAssistant

def main():
    # Initialize assistant (loads FAISS or builds it)
    assistant = ResearchAssistant(rebuild_index=False)

    print("\nRunning required project demo questions...\n")

    q1 = "What is a convolutional neural network and what is it used for?"
    q2 = "According to the documents, what are two advantages of transformers over RNNs?"
    q3 = "Summarize the main ideas about AI applications in healthcare from the loaded files."

    # Run the three required questions
    for i, q in enumerate([q1, q2, q3], start=1):
        out = assistant.ask(q)
        print(f"\nQ{i}: {q}")
        print(f"A{i}: {out['answer']}")
        print(f"Sources: {out['sources']}")
        print("=" * 80)

    print("\nDemo complete.")
    print("You can now ask additional questions interactively!")
    print("Type 'exit' to quit.\n")

    # Interactive mode
    while True:
        user_q = input("Ask a question → ")

        if user_q.lower() in ["exit", "quit", "q"]:
            print("Goodbye!")
            break

        response = assistant.ask(user_q)
        print("\nAnswer:", response["answer"])
        print("Sources:", response["sources"])
        print("-" * 80)


if __name__ == "__main__":
    main()
