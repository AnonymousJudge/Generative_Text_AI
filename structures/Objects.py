from pydantic import BaseModel


class text_data(BaseModel):
    title: str
    keywords: list[str]
    text: str