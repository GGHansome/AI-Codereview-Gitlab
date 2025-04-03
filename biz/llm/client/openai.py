import os
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

from openai import OpenAI

from biz.llm.client.base import BaseClient
from biz.llm.types import NotGiven, NOT_GIVEN

class ResearchPaperExtraction(BaseModel):
    english_response: str = Field(description="The response in English")
    chinese_response: str = Field(description="The response in Chinese")

class OpenAIClient(BaseClient):
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_API_BASE_URL", "https://api.openai.com")
        if not self.api_key:
            raise ValueError("API key is required. Please provide it or set it in the environment variables.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        self.default_model = os.getenv("OPENAI_API_MODEL", "gpt-4o-mini")
        self.provider = "openai"

    def completions(self,
                    messages: List[Dict[str, str]],
                    model: Optional[str] | NotGiven = NOT_GIVEN,
                    ) -> str:
        model = model or self.default_model
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
        )
        return completion.choices[0].message.content

    def completions_parse(self,
                          messages: List[Dict[str, str]],
                          model: Optional[str] | NotGiven = NOT_GIVEN,
                          ) -> ResearchPaperExtraction:
        model = model or self.default_model
        completion = self.client.beta.chat.completions.parse(
            model=model,
            messages=messages,
            response_format=ResearchPaperExtraction,
        )
        return completion.choices[0].message.parsed
