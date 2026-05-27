from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware 
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from arq import create_pool
import json
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from middleware import limiter, rate_limit_exceeded_handler
from ai import stream_agent_tokens_async
from config import REDIS_SETTINGS


class Query(BaseModel):
    userQuery : str
    userId : Optional[str] = None
    anonId : Optional[str] = None 



app = FastAPI()

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

redis_pool = None

@app.on_event("startup")
async def startup():
    global redis_pool
    redis_pool = await create_pool(REDIS_SETTINGS)


app.add_middleware(
		CORSMiddleware,
		allow_origins=["*"],
		allow_origin_regex=".*",
		allow_credentials=False,
		allow_methods=["*"],
		allow_headers=["*"],
	)


@app.post("/query")
async def query_stream(queryData: Query):

    is_authenticated = queryData.userId is not None
    session_id = queryData.userId or queryData.anonId

    await redis_pool.enqueue_job(
        "log_query_job", 
        {
            
            "user_id" : session_id,
            "is_authenticated" : is_authenticated,
            "query" : queryData.userQuery

        }
        )

    async def event_stream():
        try:
            async for token in stream_agent_tokens_async(queryData.userQuery, session_id):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "\nevent: done\n\n"
        except Exception as exc:
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"
  
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
