import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import AsyncGenerator

from langchain_core.messages import HumanMessage
from agent import app as agent_app

logger = logging.getLogger(__name__)

router = APIRouter()


class QuestionInput(BaseModel):
    question: str


def run_single_question(question: str) -> str:
    """Run the agent with a single question and return the final answer."""
    result = agent_app.invoke({"messages": [HumanMessage(content=question)]})
    return result["messages"][-1].content


async def stream_agent_response(question: str) -> AsyncGenerator[str, None]:
    """Yield server-sent event chunks for a given question."""
    try:
        for chunk in agent_app.stream(
            {"messages": [HumanMessage(content=question)]}, stream_mode="values"
        ):
            response = chunk["messages"][-1].content
            if response:
                yield f"data: {response}\n\n"
    except Exception as e:
        yield f"data: Error: {str(e)}\n\n"


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@router.post("/invoke")
async def invoke(input_data: QuestionInput):
    """Run the agent with a single question and return the result."""
    try:
        result = run_single_question(input_data.question)
        return {"result": result}
    except Exception as e:
        logger.exception("Error processing question: %s", input_data.question)
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/invoke-streaming")
async def invoke_streaming(input_data: QuestionInput):
    """Run the agent with a single question and stream the response."""
    try:
        return StreamingResponse(
            stream_agent_response(input_data.question),
            media_type="text/event-stream",
        )
    except Exception as e:
        logger.exception("Error streaming response for question: %s", input_data.question)
        raise HTTPException(status_code=500, detail="Internal server error")
