from langchain_google_genai import ChatGoogleGenerativeAI
from config import GOOGLE_API_KEY, GEMINI_MODEL, TEMPERATURE, MAX_OUTPUT_TOKENS

llm = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    google_api_key=GOOGLE_API_KEY,
    temperature=TEMPERATURE,
    max_output_tokens=MAX_OUTPUT_TOKENS,
)
