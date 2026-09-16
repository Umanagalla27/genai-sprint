import os
import sys
from dotenv import load_dotenv
from groq import Groq, APIError

# Load environment variables from .env file
load_dotenv()

def get_api_client() -> Groq:
    """Validate and return Groq client instance."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY is not set in your .env file.", file=sys.stderr)
        print("Create a .env file with: GROQ_API_KEY=gsk_...", file=sys.stderr)
        sys.exit(1)
    return Groq(api_key=api_key)

def generate_summary(topic: str, client: Groq, model: str = "groq/compound-mini") -> str:
    """
    Call Groq LLM API to generate a 3-bullet executive summary.
    
    Args:
        topic: The topic to summarize.
        client: The Groq client instance.
        model: Open-weight model hosted on Groq.
        
    Returns:
        3-bullet summary string.
    """
    if not topic.strip():
        raise ValueError("Topic cannot be empty.")

    system_prompt = (
        "You are an executive AI assistant. Summarize the user's topic in exactly "
        "3 concise, high-impact bullet points. Each bullet must be under 25 words."
    )
    user_prompt = f"Topic to summarize: {topic}"

    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.2,  # Low temperature for factual, deterministic output
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content if content else "No response generated."
    except APIError as e:
        return f"Groq API Error: {str(e)}"
    except Exception as e:
        return f"Unexpected failure: {str(e)}"

def main() -> None:
    """CLI Entry point."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
    else:
        topic = input("Enter a topic to summarize: ").strip()

    if not topic:
        print("Error: No topic provided.")
        sys.exit(1)

    client = get_api_client()
    print(f"\n--- Generating 3-Bullet Summary for: '{topic}' ---")
    summary = generate_summary(topic=topic, client=client)
    print("\nResult:")
    print(summary)
    print("\n-------------------------------------------------------------")

if __name__ == "__main__":
    main()