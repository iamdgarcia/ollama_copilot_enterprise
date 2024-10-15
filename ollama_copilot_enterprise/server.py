import asyncio
from typing import AsyncIterable

from fastapi import FastAPI, HTTPException

from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.schema import HumanMessage
from pydantic import BaseModel
from ollama_copilot_enterprise.code_generation import get_code_agent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageResponse(BaseModel):
    message: str

class Message(BaseModel):
    content: str

agent = get_code_agent()
@app.post("/invoke/",response_model=MessageResponse)
async def invoke(message: Message):
    solution = agent.invoke({"messages": [("user", message.content)], "iterations": 0, "error": "", "context": ""})
    print(solution["generation"])
    try:
        return MessageResponse(message=solution["generation"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", reload=True)