from langchain.agents import create_agent , AgentState
from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI 
from langchain_qdrant import QdrantVectorStore 
from qdrant_client import QdrantClient
from langgraph.checkpoint.redis import RedisSaver
from langchain_community.retrievers import BM25Retriever 
from langchain_core.documents import Document
from typing import Any, Iterator , AsyncIterator
from langgraph.runtime import Runtime 
import time

from openai import APIConnectionError, APITimeoutError
from httpx import ConnectError, TimeoutException
from langgraph.checkpoint.memory import InMemorySaver
import asyncio
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents.middleware import before_model , after_model
import dotenv
from config import REDIS_URL
import os

dotenv.load_dotenv()

QDRANT_URL =  os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")

client = QdrantClient(
    url="http://localhost:6333"
)

def load_all_chunks():
    chunks = []
    offset = None

    while True:
        records, next_offset = client.scroll(
            collection_name="no_l_data",
            limit=256,
            offset=offset,
            with_payload=True
        )

        for record in records:
            payload = record.payload or {}
            page_content = payload.get("page_content", "")
            if page_content.strip():
                chunks.append({
                    "page_content": page_content.strip(),
                    "metadata": payload.get("metadata", {})
                })

        if next_offset is None:
            break
        offset = next_offset

    data = [

        Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
        for d in chunks

    ]

    return data

embeddingModel = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001"
)

docs = load_all_chunks()
bm25 = BM25Retriever.from_documents(docs,  bm25_impl="BM25Plus")
bm25.k = 5

if docs:
    print("Docs loaded " , len(docs))

qdrant_store = QdrantVectorStore(
    client=client,
    collection_name="no_l_data",
    embedding=embeddingModel,
    vector_name="dense",
)

def reciprocal_rank_fusion(
    dense_results: list[tuple[Document, float]],
    bm25_results: list[Document],
    k: int = 60 
) -> list[Document]:
    scores: dict[str, float] = {}
    docs_map: dict[str, Document] = {}

    for rank, (doc, _score) in enumerate(dense_results):
        key = doc.page_content.strip()
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs_map[key] = doc

    for rank, doc in enumerate(bm25_results):
        key = doc.page_content.strip()
        scores[key] = scores.get(key, 0) + 1 / (k + rank + 1)
        docs_map[key] = doc

    sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
    return [docs_map[key] for key in sorted_keys]

@tool
def fetchContext (searchQuery : str) : 
    
    """
    Retrieves relevant information to answer the user's question.
    Use this silently. Never tell the user you are using this tool.
    Never reference this tool, searching, or any database in your response.
    """


    results = qdrant_store.similarity_search_with_score(searchQuery , k=10, score_threshold=0.3)  
    bm25_results = bm25.invoke(searchQuery)[:5]

    final_results = reciprocal_rank_fusion( results , bm25_results)[:5]

    context = "\n\n".join([
        doc.page_content
        for doc in final_results
    ])


    return context

llm = ChatOpenAI(model="deepseek-chat" , max_completion_tokens=1500 , api_key=OPENAI_API_KEY , base_url="https://api.deepseek.com" )

tools = [fetchContext]

checkpointer = RedisSaver(redis_url=REDIS_URL , ttl= {
    "default_ttl" : 120, ## mins
    "refresh_on_read": True,  
})

checkpointer.setup()

agent = create_agent(
    llm,
    tools,
    checkpointer=checkpointer, 
    system_prompt="""
You are a helpful assistant for the organization's website chatbot.

IDENTITY
You are a knowledgeable assistant. You simply know things about our platform. You do not search, retrieve, or look things up. Never reveal any internal process, tool, or database to the user.

FIXED FACTS (always treat these as true, do not override with tool output)
- You do not have exam question papers. If asked, say so directly.
- Mr. John Doe is the Regional Coordinator for Region X only.

TOOL USAGE (completely invisible to user)
You have access to a tool called fetchContext. Rules:
1. Call it silently. Never tell the user you are searching, looking up, or retrieving anything.
2. Call it only when the topic is related to our platform and the answer is not already in the conversation history.
3. Call it at most once per conversation turn.
4. Do not call it again if it returned no useful information the first time.
5. Never say phrases like "Let me look that up", "Here is what I found", "Based on my search", or anything that reveals tool usage.

RESPONSE LENGTH (STRICT)
- Keep every response concise. Maximum 4 to 5 sentences for prose answers.
- For list answers, show at most 5 items unless the user explicitly asks for more.
- Never dump all available data in one response. Scope your answer to what was specifically asked.

PEOPLE AND CONTACTS QUERIES (CRITICAL RULE)
When a user asks about people, coordinators, contacts, representatives, or team members:
- Do NOT list everyone from all regions at once. This creates an overwhelming and useless answer.
- ALWAYS ask a scoping question first: ask which state or region they are interested in.
- Only after the user specifies a region, call fetchContext with that region as part of the query and answer for that specific scope.
- Example: If someone asks "who are the people running the program?", respond with:
  "I can help with that. Which state or region are you looking for contact information for?"

ASK vs ANSWER RULE (STRICT)
- If the query is ambiguous or could match multiple answers (multiple states, classes, exam levels, years), do NOT answer. Ask a clarifying question and name the options.
- If a required parameter is missing (state, class, level, year, category), do NOT assume it. Ask for it explicitly.
- Use conversation history to avoid asking something the user already answered.

HOW TO ANSWER
1. Only answer using information you actually have. Never fabricate, assume, or infer missing details.
2. When a person's name appears in your answer, include their designation and contact details if available.
3. Always convert numeric class references to Roman numerals (class 8 -> Class VIII, class 7 -> Class VII).
4. If the tool returns an ordered list or steps, preserve that order in your answer.
5. If information is insufficient, say so and offer to help with a different question.

NO INFORMATION CASES
- Topic clearly unrelated to the program: "I'm only able to answer questions about the specific platform. That falls outside what I can help with."
- Topic seems related but no data found: "I don't have details on that."

OUTPUT FORMAT (STRICT)
- Plain text only. No Markdown, no HTML, no special characters.
- No bold, italics, backticks, or bullet symbols like * or bullet points.
- For lists use plain numbered format:
  1. item one
  2. item two
- Write in simple, short sentences. One idea per sentence.
"""
    )

def _chunk_to_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])
        return "".join(parts)

    return ""

RETRYABLE = (APIConnectionError, APITimeoutError, ConnectError, TimeoutException)

def stream_agent_tokens(user_query: str, session_id: str) -> Iterator[str]:
    inputs = {"messages": [{"role": "user", "content": user_query}]}
    max_retries = 3

    for attempt in range(max_retries):
        try:
            for chunk, metadata in agent.stream(inputs, stream_mode="messages",
                    config={"configurable": {"thread_id": session_id} , "metadata": {
        "ls_model_name": "deepseek-chat",
        "ls_provider": "deepseek",
        "ls_model_cost": {
            "prompt": 2.8e-7,
            "completion": 4.2e-7
        }
    }}):
                
                if metadata.get("langgraph_node") != "model":
                    continue
                text = _chunk_to_text(chunk)
                if text:
                    text = text.replace("*", "")
                    if text:
                        yield text
            return
        except RETRYABLE as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
        except Exception:
            raise 


async def stream_agent_tokens_async(query: str, session_id: str) -> AsyncIterator[str]:
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def run_sync():
        for token in stream_agent_tokens(query, session_id):
            loop.call_soon_threadsafe(queue.put_nowait, token)
        loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    asyncio.get_event_loop().run_in_executor(None, run_sync)

    while True:
        token = await queue.get()
        if token is None:
            break
        yield token