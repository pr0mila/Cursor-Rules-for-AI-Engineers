# ⚡ Cursor Rules for AI Engineers
### Stop Fighting Your Editor. Start Shipping AI Products.

A battle-tested collection of `.cursorrules` files built specifically for AI engineers — LLM apps, RAG pipelines, agents, fine-tuning workflows, and everything in between. Clone the rules that match your stack and let Cursor write code that actually fits your project.

---

## 📖 What Are Cursor Rules?

Cursor Rules are project-level instructions stored in a `.cursorrules` file at the root of your repository. They act as a persistent system prompt for Cursor's AI — defining conventions, enforcing patterns, and keeping the model grounded in your specific stack and architecture.

For AI engineers in particular, this matters a lot. LLM code is full of sharp edges: token counting, async streaming, prompt injection risks, model versioning, retry logic, embedding dimensions. Without rules, the AI guesses. With rules, it knows.

---

## ✨ Why AI Engineers Need This

- **LLM-aware code suggestions** — rules that understand prompt templates, token limits, streaming patterns, and tool/function calling conventions
- **RAG pipeline consistency** — enforce chunking strategies, embedding models, vector store patterns, and retrieval logic across your codebase
- **Agent architecture guardrails** — keep multi-agent systems structured, observable, and debuggable
- **Model-provider agnostic patterns** — switch between OpenAI, Anthropic, Cohere, or open-source models without rewriting your conventions
- **Fewer hallucinated imports** — AI stops suggesting deprecated or nonexistent SDK methods when it knows exactly which version you're using
- **Faster prototyping, cleaner handoff** — go from notebook experiments to production-grade code without losing structure

---

## 🚀 Setup

### 1. Clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/cursor-rules-ai.git
cd cursor-rules-ai
```

### 2. Pick the template that matches your stack

```bash
cp rules/langchain-python/.cursorrules /path/to/your-project/.cursorrules
```

### 3. Open your project in Cursor

Cursor automatically picks up `.cursorrules` from the project root. No config needed — it applies immediately.

### 4. Tailor it to your project

Add your specific model names, vector DB of choice, embedding dimensions, prompt formats, and any team conventions. The more specific, the better the output.

---

## 📁 Project Structure

```
cursor-rules-ai/
├── README.md
├── rules/
│   ├── langchain-python/
│   │   └── .cursorrules
│   ├── llm-api-integration/
│   │   └── .cursorrules
│   ├── rag-pipeline/
│   │   └── .cursorrules
│   ├── ai-agents/
│   │   └── .cursorrules
│   ├── fine-tuning-workflows/
│   │   └── .cursorrules
│   └── fastapi-llm-backend/
│       └── .cursorrules
└── CONTRIBUTING.md
```

---

## 📋 Templates

---

### 🦜 LangChain + Python

```
You are an expert AI engineer working with LangChain (v0.2+), Python 3.11+, and LLM application development.

LangChain Patterns:
- Use LangChain Expression Language (LCEL) for all chain construction — pipe operators over legacy LLMChain
- Prefer RunnableParallel, RunnableLambda, and RunnablePassthrough for composability
- Use ChatPromptTemplate.from_messages() for all prompt construction
- Always set model names explicitly — never rely on defaults
- Use callbacks for logging and tracing (LangSmith preferred)

LLM Usage:
- Always pass temperature and max_tokens explicitly
- Use structured outputs (with_structured_output()) over parsing raw strings
- Handle rate limits with tenacity retry decorators
- Never log raw prompt contents in production — they may contain PII

Memory & State:
- Use ConversationBufferWindowMemory with a defined k to cap token usage
- Prefer external memory stores (Redis, PostgreSQL) over in-process for production
- Always serialize/deserialize memory explicitly — don't trust implicit state

RAG:
- Use RecursiveCharacterTextSplitter with explicit chunk_size and chunk_overlap
- Always embed and retrieve with the same model — document this clearly
- Return source documents with every retrieval result for traceability

Error Handling:
- Catch and handle OutputParserException explicitly
- Wrap LLM calls in try/except — APIs fail, models timeout
- Log token usage from response metadata for cost tracking

Do not use deprecated chain types (SequentialChain, SimpleSequentialChain). Avoid LangChain v0.1 patterns.
```

---

### 🔌 LLM API Integration (OpenAI / Anthropic / Multi-provider)

```
You are an expert AI engineer building production LLM integrations.

API Usage:
- Always specify model, temperature, and max_tokens explicitly in every call
- Use streaming (stream=True) for user-facing responses — never block on full completion
- Implement exponential backoff with jitter for all API calls (use tenacity or backoff library)
- Track and log token usage from every response for cost monitoring

OpenAI SDK (v1+):
- Use the openai.AsyncOpenAI client for async contexts
- Use response_format={"type": "json_object"} when expecting structured output
- Use tool_choice and tools array for function calling — not the deprecated functions param
- Always validate tool call arguments with Pydantic before executing

Anthropic SDK:
- Use anthropic.AsyncAnthropic for async contexts
- Pass system prompt as the system= parameter, not inside messages
- Use max_tokens carefully — Anthropic requires it explicitly
- Use tool_use content blocks for structured outputs

Prompt Engineering:
- Store prompt templates in versioned .txt or .yaml files, not hardcoded strings
- Always include a system prompt — never rely on model defaults
- Inject dynamic values via template variables, never f-string concatenation in LLM calls
- Document expected input/output format for every prompt template

Security:
- Validate and sanitize all user input before inserting into prompts
- Never expose raw API errors to end users — they can leak system prompt fragments
- Implement output filtering for PII, toxic content, and off-topic responses
- Never hardcode API keys — use environment variables or a secrets manager

Do not use deprecated completions endpoint. Always use chat completions or messages API.
```

---

### 🗂️ RAG Pipeline

```
You are an expert AI engineer building production RAG (Retrieval-Augmented Generation) systems.

Document Processing:
- Use RecursiveCharacterTextSplitter as the default; switch to SemanticChunker for domain-specific content
- Always set chunk_size and chunk_overlap explicitly — document the reasoning
- Preserve document metadata (source, page, section) through every pipeline stage
- Clean and normalize text before chunking: remove headers/footers, fix encoding issues

Embeddings:
- Always use the same embedding model for indexing and querying — enforce this with a config constant
- Store embedding model name and dimension in your vector DB metadata
- Batch embed documents — never embed one at a time in production
- Cache embeddings for static documents to reduce cost and latency

Vector Store (Pinecone / Weaviate / pgvector / Chroma):
- Namespace or partition by data source or tenant
- Always include metadata filters in retrieval to reduce noise
- Log retrieval scores — set a minimum similarity threshold before passing to LLM
- Use upsert, not insert, to handle document updates gracefully

Retrieval:
- Retrieve more documents than needed (top-k = 10), then rerank to top 3-5
- Use a CrossEncoder reranker (e.g., Cohere Rerank or sentence-transformers) for precision-critical use cases
- Always return source metadata alongside retrieved chunks
- Implement hybrid search (semantic + keyword) for better coverage

Generation:
- Instruct the model to answer only from provided context — include this in every system prompt
- Ask the model to cite sources using chunk IDs or document names
- Detect and handle "I don't know" gracefully — don't let the model hallucinate beyond context
- Keep retrieved context within 60-70% of the model's context window to leave room for the response

Evaluation:
- Track retrieval metrics: recall@k, MRR, NDCG
- Track generation metrics: faithfulness, answer relevance, context precision (use RAGAS)
- Log every query, retrieved chunks, and response for offline evaluation

Do not pass raw documents without chunking. Never skip metadata in retrieval results.
```

---

### 🤖 AI Agents

```
You are an expert AI engineer building LLM-powered agents and multi-agent systems.

Agent Design:
- Define a clear agent loop: observe → plan → act → reflect
- Every agent must have an explicit system prompt defining its role, capabilities, and boundaries
- Use structured tool schemas (JSON Schema / Pydantic) — never let agents call freeform strings
- Agents should return structured outputs that downstream agents or systems can parse reliably

Tool Use:
- Every tool must have: name, description, input schema, output schema, and error handling
- Validate tool inputs before execution — treat every LLM-generated argument as untrusted
- Implement timeouts for every tool call — agents must not hang indefinitely
- Log every tool call and result for observability and debugging
- Return errors to the agent in structured form — let it retry intelligently

Multi-Agent Systems:
- Define clear roles: orchestrator vs. worker agents
- Use message passing with explicit typed schemas between agents
- Avoid shared mutable state — each agent should be stateless where possible
- Implement circuit breakers to prevent cascading failures across agents

Memory:
- Separate working memory (current task context) from long-term memory (persistent store)
- Summarize and compress conversation history before it fills the context window
- Use semantic search over long-term memory — don't dump everything into the prompt

Safety & Control:
- Implement a human-in-the-loop checkpoint for high-stakes or irreversible actions
- Set a maximum iteration limit for every agent loop — prevent infinite loops
- Log the full reasoning trace (chain of thought) for every agent decision
- Never allow agents to modify their own system prompt or tool definitions at runtime

Observability:
- Trace every agent run end-to-end (use LangSmith, Langfuse, or OpenTelemetry)
- Emit structured logs: agent_id, run_id, step, tool_called, tokens_used, latency
- Alert on high token usage, tool failures, and loops exceeding N steps

Do not build agents without iteration limits, tool validation, or observability.
```

---

### 🎯 Fine-Tuning Workflows

```
You are an expert AI engineer working on LLM fine-tuning pipelines and dataset preparation.

Dataset Preparation:
- Store training data in JSONL format with explicit schema documentation
- Every example must have: instruction, input (optional), and output fields
- Validate dataset quality before training: check for duplicates, length outliers, label noise
- Split data into train/validation/test — never fine-tune and evaluate on the same set
- Log dataset statistics: size, average token count, class distribution

Data Quality:
- Use a quality scoring pipeline before adding examples to the training set
- Prefer 1,000 high-quality examples over 10,000 noisy ones
- Annotator disagreement > 20% on a sample is a signal to revisit your task definition
- Remove examples that contain PII, copyrighted content, or adversarial inputs

Training:
- Document every training run: base model, dataset version, hyperparameters, infrastructure
- Use LoRA / QLoRA for parameter-efficient fine-tuning unless full fine-tuning is justified
- Monitor training loss and eval loss together — divergence signals overfitting
- Save checkpoints — never rely on a single final model artifact

Evaluation:
- Define task-specific metrics before training, not after
- Evaluate fine-tuned model against base model on a held-out benchmark
- Run red-teaming and safety evaluations before deploying a fine-tuned model
- Test for capability regression on tasks outside the fine-tuning domain

Deployment:
- Version models with semantic versioning — model-name-v1.2.0
- Store model cards alongside artifacts: training data, intended use, limitations, eval results
- Shadow-test new model versions before full rollout — compare outputs to current production model

Do not skip eval. Do not train without versioning datasets and model artifacts.
```

---

### ⚡ FastAPI LLM Backend

```
You are an expert AI engineer building production FastAPI backends for LLM-powered applications.

Architecture:
- Organize by feature domain: /chat, /rag, /agents, /embeddings
- Keep LLM logic in service classes — never in route handlers
- Use dependency injection (Depends) for LLM clients, vector stores, and auth

Async & Streaming:
- All LLM calls must be async — use async clients (AsyncOpenAI, AsyncAnthropic)
- Stream responses using StreamingResponse with async generators
- Never block the event loop with synchronous LLM or DB calls

Request/Response:
- Validate all inputs with Pydantic v2 models — be explicit about field types and constraints
- Return consistent response envelopes: { data, error, usage, request_id }
- Include a unique request_id in every response for traceability
- Never expose raw LLM errors to clients — map them to user-friendly messages

Token & Cost Management:
- Count tokens before sending to the LLM — reject requests that exceed your limit
- Log token usage (prompt, completion, total) from every response
- Implement per-user rate limiting on token consumption, not just request count

Background Tasks:
- Use FastAPI BackgroundTasks for fire-and-forget work (logging, analytics, webhooks)
- Use Celery or ARQ for long-running jobs (embedding pipelines, batch inference)
- Never run embedding or fine-tuning jobs in the request/response cycle

Security:
- Authenticate every endpoint — use JWT or API keys via Depends
- Validate and sanitize user content before injecting into prompts
- Implement output content filtering before returning LLM responses to users
- Rate limit all public-facing endpoints (slowapi or similar)

Observability:
- Log every LLM request: model, tokens, latency, user_id, request_id
- Emit metrics: p50/p95/p99 latency, error rate, token throughput
- Add /health and /ready endpoints for infrastructure monitoring

Do not put business logic in route handlers. Do not use sync LLM clients in async routes.
```

---

## 🤝 Contributing

Have a rule set for a stack not listed here — LlamaIndex, CrewAI, AutoGen, DSPy, Haystack, or anything else in the AI engineering world? PRs are welcome.

1. Fork the repo
2. Create a folder under `rules/your-stack-name/`
3. Add a `.cursorrules` file — keep rules specific, actionable, and grounded in real patterns
4. Open a pull request with a short description of the stack and what problems the rules solve

Low-quality, vague, or filler rules won't be merged. Quality over quantity.

---

## 📄 License

MIT — use freely, fork liberally, share widely.

---

## ⭐ Star This Repo

If this saves you time, a star helps other AI engineers find it.
