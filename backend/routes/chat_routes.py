from fastapi import APIRouter, HTTPException
from backend.schema.models import ChatRequest, ChatResponse
from typing import Dict
from backend.utils.chat import generate_response  # Import the generate_response function

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> Dict[str, str]:
    try:
        session_id = "example_session"  # You might want to generate unique session IDs for each user
        response = generate_response(session_id, request.message)
        return {"response": response["response"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))