from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import JSONResponse

def get_session_id(request):
    # Rate limit per session ID, fallback to IP
    return request.headers.get("x-session-id") or get_remote_address(request)

limiter = Limiter(key_func=get_session_id)

def rate_limit_exceeded_handler(request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "Daily limit reached.",
            "message": f"You have reached your daily limit of 20 questions. Please come back tomorrow.",
        }
    )