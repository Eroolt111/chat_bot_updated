import logging
import sys
from .pipeline import ChatbotPipeline
from .config import config

logger = logging.getLogger(__name__)
 
def main():
    logging.basicConfig(level=config.LOG_LEVEL)
    pipeline = ChatbotPipeline()
    print("Chatbot CLI. Type 'exit' or 'quit' to stop.")
    while True: 
        try:
            question = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.") 
            break
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        try:
            result = pipeline.query_pipeline.run(input = question)
            print(f"Bot: {result}")
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            print(f"Error: {e}")
    sys.exit(0)

if __name__ == "__main__":
    main()
