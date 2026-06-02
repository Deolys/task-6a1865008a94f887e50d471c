# main.py
import os
from dotenv import load_dotenv
from typing import List, Optional

# LangChain imports
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import argparse

load_dotenv()

# Define Pydantic models with field descriptions
class PersonInfo(BaseModel):
    name: str = Field(..., description="Full name of the person")
    age: Optional[int] = Field(None, description="Age in years, optional")
    profession: str = Field(..., description="Job title or profession")
    skills: List[str] = Field(..., description="List of professional skills")

class MeetingNotes(BaseModel):
    date: str = Field(..., description="Date of the meeting in ISO format (YYYY-MM-DD)")
    participants: List[str] = Field(..., description="Names of participants")
    topics: List[str] = Field(..., description="Main discussion topics")
    decisions: List[str] = Field(..., description="Decisions made during the meeting")
    next_steps: List[str] = Field(..., description="Action items for follow‑up")

# LLM setup (OpenAI)
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

# Prompt template with format instructions placeholder
prompt_template = PromptTemplate(
    input_variables=["text", "format_instructions"],
    template="""
You are an assistant that extracts structured data from a single paragraph of text.
The output must be in JSON format following the provided schema.
{format_instructions}

Text: {text}
"""
)

# Heuristic to choose schema based on keywords
MEETING_KEYWORDS = [
    "встреча", "meeting", "собрание", "присутствовали",
    "темы", "решения", "next steps", "action items"
]

def is_meeting(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in MEETING_KEYWORDS)

# Main extraction function
def extract_structured(text: str):
    if is_meeting(text):
        parser = PydanticOutputParser(pydantic_object=MeetingNotes)
        schema_name = "MeetingNotes"
    else:
        parser = PydanticOutputParser(pydantic_object=PersonInfo)
        schema_name = "PersonInfo"

    partial_variables = {"format_instructions": parser.get_format_instructions()}
    prompt = prompt_template.partial(**partial_variables)
    chain = prompt | llm | parser
    result = chain.invoke({"text": text})
    return schema_name, result

# CLI handling
if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Extract structured data from a paragraph.")
    parser_cli.add_argument("--text", type=str, help="Input text to parse. If omitted, examples are used.")
    args = parser_cli.parse_args()

    if args.text:
        texts = [args.text]
    else:
        # Two example paragraphs
        texts = [
            "Анна, 28 лет, Python-разработчик. Навыки: FastAPI, Docker.",
            "Встреча 2026-05-27 с участниками Иван и Мария. Темы: проект X, бюджет. Решения: увеличить срок на 2 недели. Next steps: подготовить план." 
        ]

    for txt in texts:
        schema_name, data = extract_structured(txt)
        print(f"\nSchema: {schema_name}\n")
        print(data.model_dump(indent=2))
