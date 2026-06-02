#!/usr/bin/env python3
"""
Structured output extraction using LangChain and Pydantic.
Author: Auto-generated per instructor feedback.
"""
import os
from pathlib import Path
import argparse
from dotenv import load_dotenv

# Load environment variables (e.g., OPENAI_API_KEY)
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
import re

# ---------- Pydantic models ----------
class PersonInfo(BaseModel):
    name: str = Field(..., description="Full name of the person")
    age: int | None = Field(None, description="Age in years, optional")
    profession: str = Field(..., description="Primary occupation or role")
    skills: list[str] = Field(..., description="List of professional skills")

class MeetingNotes(BaseModel):
    date: str = Field(..., description="Meeting date in ISO format (YYYY-MM-DD)")
    participants: list[str] = Field(..., description="Names of attendees")
    topics: list[str] = Field(..., description="Discussion topics covered during the meeting")
    decisions: list[str] = Field(..., description="Decisions made in the meeting")
    next_steps: list[str] = Field(..., description="Action items to be taken after the meeting")

# ---------- Helper functions ----------
def is_meeting(text: str) -> bool:
    """Heuristic to decide if text describes a meeting."""
    keywords = ["meeting", "встреча", "дата", "темы", "решения", "next steps"]
    return any(kw.lower() in text.lower() for kw in keywords)

# ---------- LangChain setup ----------
def build_chain(model: BaseModel):
    parser = PydanticOutputParser(pydantic_object=model)
    prompt_template = PromptTemplate(
        input_variables=["text", "format_instructions"],
        template="""
You are an assistant that extracts structured data from the following text.

Text: {text}

The output must be a JSON object matching the provided schema. Follow these instructions:
{format_instructions}

Respond with only the JSON object.
""",
    )
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt_template | llm | parser
    return chain, parser

# ---------- Main logic ----------
def parse_text(text: str):
    if is_meeting(text):
        model_cls = MeetingNotes
    else:
        model_cls = PersonInfo
    chain, parser = build_chain(model_cls)
    result = chain.invoke({"text": text, "format_instructions": parser.get_format_instructions()})
    return result

# ---------- CLI ----------
if __name__ == "__main__":
    parser_cli = argparse.ArgumentParser(description="Extract structured data from text.")
    group = parser_cli.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--text", type=str, help="Input text to parse")
    group.add_argument("-e", "--example", action='store_true', help="Run with built‑in examples")
    args = parser_cli.parse_args()

    if args.example:
        examples = {
            "person": "Анна, 28 лет, Python-разработчик. Навыки: FastAPI, Docker.",
            "meeting": "Дата: 2026-06-01\nУчастники: Иван, Мария\nТемы: Архитектура, CI/CD\nРешения: Перенести в облако\nNext steps: Создать план миграции",
        }
        for key, txt in examples.items():
            print(f"\n--- Example: {key} ---")
            parsed = parse_text(txt)
            print(parsed.model_dump_json(indent=2))
    else:
        parsed = parse_text(args.text)
        print(parsed.model_dump_json(indent=2))
