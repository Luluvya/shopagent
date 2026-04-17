from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, httpx, asyncio, json, hashlib
from datetime import datetime
from typing import TypedDict, Annotated

import chromadb
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage

app = FastAPI(title="ShopAgent API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

SILICONFLOW_KEY = os.getenv("SILICONFLOW_KEY", "")
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
EMBED_URL = "https://api.siliconflow.cn/v1/embeddings"
LLM_MODEL = "deepseek-ai/DeepSeek-V3"
EMBED_MODEL = "BAAI/bge-m3"

# ── 缓存 ──────────────────────────────────────────────────
response_cache: dict = {}
CACHE_MAX = 200

def cache_get(msg: str):
    key = hashlib.md5(msg.strip().lower().encode()).hexdigest()
    if key in response_cache:
        response_cache[key]["hits"] += 1
        return response_cache[key]["data"]
    return None

def cache_set(msg: str, data: dict):
    if len(response_cache) >= CACHE_MAX:
        oldest = min(response_cache, key=lambda k: response_cache[k]["ts"])
        del response_cache[oldest]
    key = hashlib.md5(msg.strip().lower().encode()).hexdigest()
    response_cache[key] = {"data": data, "ts": datetime.now().isoformat(), "hits": 0}

# ── ChromaDB ──────────────────────────────────────────────
chroma_client = chromadb.Client()
shop_collection = chroma_client.get_or_create_collection("shopee_knowledge")
kb_version = {"version": 1, "updated_at": datetime.now().isoformat()}

# ── Embedding ─────────────────────────────────────────────
async def get_embedding(text: str) -> list:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            EMBED_URL,
            headers={"Authorization": f"Bearer {SILICONFLOW_KEY}"},
            json={"model": EMBED_MODEL, "input": text[:500]}
        )
        return resp.json()["data"][0]["embedding"]

# ── LLM ───────────────────────────────────────────────────
async def llm(system: str, user: str, max_tokens: int = 1000) -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            API_URL,
            headers={"Authorization": f"Bearer {SILICONFLOW_KEY}", "Content-Type": "application/json"},
            json={
                "model": LLM_MODEL,
                "max_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ]
            }
        )
        return resp.json()["choices"][0]["message"]["content"]

# ── RAG ───────────────────────────────────────────────────
async def retrieve(query: str, n: int = 3) -> str:
    if shop_collection.count() == 0:
        return ""
    emb = await get_embedding(query)
    results = shop_collection.query(
        query_embeddings=[emb],
        n_results=min(n, shop_collection.count())
    )
    docs = results.get("documents", [[]])[0]
    return "\n\n---\n\n".join(docs) if docs else ""

# ── Init RAG ──────────────────────────────────────────────
async def init_rag():
    data_dir = "data"
    if not os.path.exists(data_dir):
        return
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if not files or shop_collection.count() > 0:
        print(f"RAG ready: {shop_collection.count()} chunks")
        return
    idx = 0
    for fname in files:
        with open(os.path.join(data_dir, fname), "r", encoding="utf-8") as f:
            content = f.read()
        chunks = [c.strip() for c in content.split("---") if len(c.strip()) > 50]
        for chunk in chunks:
            emb = await get_embedding(chunk)
            shop_collection.add(documents=[chunk], embeddings=[emb], ids=[f"doc_{idx}"])
            idx += 1
    kb_version["updated_at"] = datetime.now().isoformat()
    print(f"RAG initialized: {shop_collection.count()} chunks")

# ── Agent State ───────────────────────────────────────────
class ShopState(TypedDict):
    messages: Annotated[list, add_messages]
    intent: str
    user_input: str
    product_result: str
    pricing_result: str
    logistics_result: str
    risk_result: str
    general_result: str
    final_response: str
    has_rag: bool
    history: list

# ── Supervisor ────────────────────────────────────────────
async def supervisor_node(state: ShopState) -> ShopState:
    history_str = "\n".join([
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:60]}"
        for m in state.get("history", [])[-4:]
    ])
    intent = await llm(
        """你是跨境电商选品助手的任务调度器。
判断用户意图，只返回以下之一，不要其他文字：
product    - 选品建议、热销品类、什么产品好卖
pricing    - 定价策略、利润计算、成本核算
logistics  - 物流方案、运费、时效、发货方式
risk       - 风险提示、违禁品、合规、关税政策
general    - 打招呼、其他问题""",
        f"历史：{history_str}\n用户：{state['user_input']}"
    )
    intent = intent.strip().lower()
    if intent not in ["product", "pricing", "logistics", "risk"]:
        intent = "general"
    return {**state, "intent": intent}

# ── Product Agent ─────────────────────────────────────────
async def product_node(state: ShopState) -> ShopState:
    context = await retrieve(state["user_input"] + " 热销品类 利润")
    sys = f"""你是专业的 Shopee 东南亚跨境电商选品顾问。
{'参考知识库：\n' + context if context else ''}
根据用户需求推荐适合的选品方向，包含：
## 推荐品类
品类名称、热销程度（⭐评级）、平均利润率、主要市场

## 选品理由
为什么适合、目标用户群体、市场需求分析

## 进货建议
推荐货源渠道、参考进货价区间

## 竞争分析
竞争程度、差异化建议

用专业友好的中文回答，数据具体，有参考价值。"""
    result = await llm(sys, state["user_input"])
    return {**state, "product_result": result, "has_rag": bool(context)}

# ── Pricing Agent ─────────────────────────────────────────
async def pricing_node(state: ShopState) -> ShopState:
    context = await retrieve(state["user_input"] + " 定价 成本 利润")
    sys = f"""你是跨境电商定价策略专家。
{'参考知识库：\n' + context if context else ''}
根据用户描述给出定价建议，包含：
## 成本拆解
进货成本、头程运费、平台佣金（约3-7%）、推广费（约5-8%）、退货损耗

## 建议定价
低/中/高客单价方案，各方案的利润率

## 定价策略
新店冲量定价 vs 成熟店利润定价，如何参考竞品定价

数据要具体，给出计算示例。"""
    result = await llm(sys, state["user_input"])
    return {**state, "pricing_result": result, "has_rag": bool(context)}

# ── Logistics Agent ───────────────────────────────────────
async def logistics_node(state: ShopState) -> ShopState:
    context = await retrieve(state["user_input"] + " 物流 运费 海外仓")
    sys = f"""你是跨境电商物流专家，专注东南亚市场。
{'参考知识库：\n' + context if context else ''}
根据用户需求推荐物流方案，包含：
## 推荐方案
直邮 vs 海外仓的对比，适用场景

## 物流商推荐
具体物流商名称、优缺点、适合的目的地

## 费用参考
头程运费区间、时效、注意事项

## 操作建议
新手建议先直邮测款，爆款再转海外仓。"""
    result = await llm(sys, state["user_input"])
    return {**state, "logistics_result": result, "has_rag": bool(context)}

# ── Risk Agent ────────────────────────────────────────────
async def risk_node(state: ShopState) -> ShopState:
    context = await retrieve(state["user_input"] + " 风险 禁售 关税 合规")
    sys = f"""你是跨境电商合规风险专家。
{'参考知识库：\n' + context if context else ''}
根据用户描述给出风险提示，包含：
## 平台规则风险
是否涉及禁售品类、侵权风险、违规操作

## 各国关税政策
目标市场的免税额度、需缴纳的税率

## 合规要求
需要的认证（如FDA、CE、SGS等）、注意事项

## 规避建议
如何降低风险、合法合规经营建议

⚠️ 风险提示要具体，帮用户避免踩坑。"""
    result = await llm(sys, state["user_input"])
    return {**state, "risk_result": result, "has_rag": bool(context)}

# ── General Agent ─────────────────────────────────────────
async def general_node(state: ShopState) -> ShopState:
    history_str = "\n".join([
        f"{'用户' if m['role']=='user' else '助手'}：{m['content'][:100]}"
        for m in state.get("history", [])[-6:]
    ])
    result = await llm(
        """你是 ShopAgent，专注东南亚跨境电商的 AI 选品决策助手。
能帮助用户：选品推荐、定价策略、物流方案、合规风险分析。
友好专业地回答，适当用 emoji，不超过150字。""",
        f"历史：\n{history_str}\n用户：{state['user_input']}"
    )
    return {**state, "general_result": result, "has_rag": False}

# ── Synthesis ─────────────────────────────────────────────
async def synthesis_node(state: ShopState) -> ShopState:
    response = (
        state.get("product_result") or
        state.get("pricing_result") or
        state.get("logistics_result") or
        state.get("risk_result") or
        state.get("general_result") or
        "抱歉，我暂时无法回答这个问题。"
    )
    return {**state, "final_response": response}

# ── Build Graph ───────────────────────────────────────────
def build_graph():
    g = StateGraph(ShopState)
    g.add_node("supervisor", supervisor_node)
    g.add_node("product",    product_node)
    g.add_node("pricing",    pricing_node)
    g.add_node("logistics",  logistics_node)
    g.add_node("risk",       risk_node)
    g.add_node("general",    general_node)
    g.add_node("synthesis",  synthesis_node)
    g.set_entry_point("supervisor")
    g.add_conditional_edges("supervisor",
        lambda s: s.get("intent", "general"),
        {"product":"product","pricing":"pricing","logistics":"logistics","risk":"risk","general":"general"}
    )
    for node in ["product","pricing","logistics","risk","general"]:
        g.add_edge(node, "synthesis")
    g.add_edge("synthesis", END)
    return g.compile()

graph = build_graph()

# ── Startup ───────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    await init_rag()

# ── Endpoints ─────────────────────────────────────────────
AGENT_LABELS = {
    "product":   "🛍 选品 Agent",
    "pricing":   "💰 定价 Agent",
    "logistics": "🚚 物流 Agent",
    "risk":      "⚠️ 风险 Agent",
    "general":   "🤖 Supervisor",
}

class ChatRequest(BaseModel):
    message: str
    history: list = []

@app.post("/chat")
async def chat(req: ChatRequest):
    # 缓存命中
    if not req.history:
        cached = cache_get(req.message)
        if cached:
            return {**cached, "from_cache": True}

    state: ShopState = {
        "messages": [HumanMessage(content=req.message)],
        "intent": "", "user_input": req.message,
        "product_result": "", "pricing_result": "",
        "logistics_result": "", "risk_result": "",
        "general_result": "", "final_response": "",
        "has_rag": False, "history": req.history,
    }
    result = await graph.ainvoke(state)
    intent = result.get("intent", "general")
    data = {
        "response": result.get("final_response", ""),
        "intent": intent,
        "agent": AGENT_LABELS.get(intent, "🤖"),
        "has_rag": result.get("has_rag", False),
        "from_cache": False
    }
    if not req.history:
        cache_set(req.message, data)
    return data

@app.post("/upload-knowledge")
async def upload(file: UploadFile = File(...)):
    content = (await file.read()).decode("utf-8")
    chunks = [c.strip() for c in content.split("---") if len(c.strip()) > 50]
    count = shop_collection.count()
    for i, chunk in enumerate(chunks):
        emb = await get_embedding(chunk)
        shop_collection.add(documents=[chunk], embeddings=[emb], ids=[f"doc_{count+i}"])
    kb_version["version"] += 1
    kb_version["updated_at"] = datetime.now().isoformat()
    response_cache.clear()
    return {"added": len(chunks), "total": shop_collection.count(), "kb_version": kb_version["version"]}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "knowledge_chunks": shop_collection.count(),
        "kb_version": kb_version["version"],
        "cache_size": len(response_cache),
        "model": LLM_MODEL
    }
