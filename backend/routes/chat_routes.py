from fastapi import APIRouter, HTTPException, Request
from backend.schema.models import ChatRequest, ChatResponse
from typing import Dict
from backend.utils.chat import generate_response  # Import the generate_response function

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, chat_request: ChatRequest) -> Dict[str, str]:
    limiter = request.app.state.limiter
    @limiter.limit("500/minute")
    async def rate_limited_chat(request: Request):
        try:
            session_id = "example_session"  # You might want to generate unique session IDs for each user
            response = generate_response(session_id, chat_request.message)
            return {"response": response["response"]}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    return await rate_limited_chat(request)