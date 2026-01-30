import getpass
import os
from typing import TypedDict, Annotated

from langchain_groq import ChatGroq

# Store the API key temporarily
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Initialize the Groq chat model
model = ChatGroq(model="llama-3.3-70b-versatile")

# Define output schema
class Review(TypedDict):
    summary: Annotated[list[str], "Brief summary points of the given review"]
    sentiment: str

# Enable structured output
structured_model = model.with_structured_output(Review)

# Invoke the model by using default long txt 
result = structured_model.invoke("""
The OnePlus 15 is a top-tier Android flagship offering, highlighted by exceptional battery life
that exceeds many competitors, lightning-fast performance, and a sleek design.
It directly challenges Samsung and Google’s best with a balanced mix of cutting-edge specs
and AI features, all while maintaining a more competitive price point.
""")

print(result)
