You can implement this as a multi-agent credit-decisioning “loan orchestration” system in ADK, with a root Orchestrator agent that coordinates specialized sub‑agents and tools, and uses ADK session state as the per‑application scratchpad while keeping the platform stateless for PII beyond each session. [google.github](https://google.github.io/adk-docs/sessions/state/)

## High‑level agent topology

- OrchestratorAgent (root, LlmAgent / Flow controller)  
- DataCollectionAgent (handles application intake & validation)  
- CreditAndFraudAgent (orchestrates Credit Bureau, Fraud API, Bank Statement API, Employment API)  
- PolicyReasoningAgent (policy QA over Confluence, applies latest rules)  
- RiskScoringAgent (wraps internal ML risk model + historical decisions lookups)  
- DecisionAndReportingAgent (turns all signals into a structured, explainable risk report + decision)  
- EscalationAgent (prepares human-underwriter package and hands off)

All of these are BaseAgent instances wired into a Multi‑Agent System pattern, sharing a single ADK Session for the application while keeping PII in volatile state only. [google.github](https://google.github.io/adk-docs/agents/multi-agents/)

***

## Sub‑agents and tool definitions

### 1) OrchestratorAgent

**Role:** Entry point per loan application; coordinates flow, error handling, and routing (auto‑decision vs human escalation).

**Key responsibilities:**

- Create one ADK Session per application, keyed by application_id (no cross‑session sharing).  
- Initialize `session.state` with non‑PII control fields: `status`, `decision_channel`, `api_call_counts`, `errors`, `latency_budget_ms`, `is_straightforward_candidate`. [codelabs.developers.google](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/3-developing-agents/build-a-multi-agent-system-with-adk)
- Decide execution plan:
  - Fire off CreditAndFraudAgent early (parallel external calls).  
  - Trigger PolicyReasoningAgent and RiskScoringAgent when raw data is ready.  
  - Decide “auto‑approve/deny vs escalate” using policy + risk thresholds and policy flags.

**Tool(s):**

- `log_event_tool`: Append structured events (who/what/when/why) to an audit log store with a correlation_id; only non‑sensitive metadata (e.g., “credit_bureau_call_started”, “policy_version=2026‑W06”).  
- `rate_limit_counter_tool`: Atomically increment/check per‑API counters (e.g., Redis/Cloud Memorystore or Firestore) for rate limiting (credit bureau 100 calls/min).  
- `notify_underwriter_tool`: Send work item + report ID to case management (e.g., Pub/Sub, Firestore task, or ticketing API).

The Orchestrator itself usually has no direct PII tools; it passes opaque identifiers to PII‑handling tools and keeps business logic and orchestration in instructions.

***

### 2) DataCollectionAgent

**Role:** Validate and normalize the inbound application payload; ensure completeness before hitting expensive tools.

**Responsibilities:**

- Validate: income present, employer name, requested amount, consent flags, missing documents.  
- If missing artifacts (e.g., bank statements), generate tasks back to the front end (but not needed if everything comes pre‑validated).  
- Normalize fields (currency, income frequency, employment type).

**Tools:**

- `schema_validation_tool`: Pure computation tool validating JSON against an internal schema, returning normalized structure + error codes.  
- `document_manifest_tool`: For uploaded documents, returns file handles (not contents) that downstream tools (Bank Statement API) can use.

**State usage (session.state):** Store a sanitized, normalized view: `application_summary` (no full SSN, card PAN, etc.; only tokens/handles referencing secure vaults). ADK state acts like a scratchpad for this session only and is not a system‑wide memory. [cloud.google](https://cloud.google.com/blog/topics/developers-practitioners/remember-this-agent-state-and-memory-with-adk)

***

### 3) CreditAndFraudAgent

**Role:** Single agent responsible for all external risk data; executes calls in parallel where safe and applies rate limiting and retries.

**Responsibilities:**

- Call Credit Bureau, Fraud Detection, Employment Verification, and Bank Statement APIs concurrently, within the latency budget.  
- Manage async Employment Verification callbacks: either:
  - use LoopAgent pattern to poll a status service, or  
  - integrate with a durable external orchestrator (e.g., Restate, Workflows) and resume the ADK session later. [github](https://github.com/restatedev/restate-google-adk-example)
- Perform exponential‑backoff and partial‑failure logic; downgrade gracefully when some APIs fail (e.g., use bank statements + internal behavior if bureau is down).

**Tools:**

- `credit_bureau_tool`  
  - Input: `application_id`, `credit_ref_token` (points to encrypted PII in a vault), minimal context.  
  - Output: score, delinquency flags, utilization, inquiries.  
  - Wrapped with internal rate‑limiting middleware that uses `rate_limit_counter_tool` and returns `THROTTLED` vs `HARD_FAILURE`.

- `fraud_detection_tool`  
  - Input: `application_id`, KYC identifiers (tokens).  
  - Output: fraud score, rules fired.

- `employment_verification_tool`  
  - Supports async: returns `REQUEST_ACCEPTED` + `verification_request_id` first; later a callback populates result into a secure store, which the agent polls via `get_verification_status_tool`.

- `bank_statement_analysis_tool`  
  - Input: document handle(s).  
  - Output: derived income, stability metrics, anomalies.

**Parallelization:** Within ADK, you can model this either as:  

- A single LlmAgent issuing several tool calls in one reasoning step (ADK supports multi‑tool selection), or  
- A small MAS: a sub‑orchestrator with parallel sub‑agents (one per API) coordinated with a LoopAgent that waits until all results or timeouts appear in state. [google.github](https://google.github.io/adk-docs/agents/multi-agents/)

**State:** Writes `session.state['credit_data']`, `['fraud_signals']`, `['employment_data']`, `['bank_income_summary']`. This is all ephemeral per session; nothing persisted beyond what audit logging requires (which should exclude raw PII). [google.github](https://google.github.io/adk-docs/sessions/state/)

***

### 4) PolicyReasoningAgent

**Role:** Read the latest policy documents from Confluence and turn them into explicit, traceable decisions for this application.

**Responsibilities:**

- Retrieve most recent policy snapshots (weekly updates) for:
  - eligibility rules (min score, DTI thresholds, prohibited industries),  
  - special handling: borderline definitions, manual review triggers, segment‑specific conditions.  
- Answer questions like:
  - “Is this application straightforward according to current policies?”  
  - “What policy sections apply to this applicant profile?”  
- Produce an intermediate “policy verdict” object referencing explicit policy sections for explainability.

**Tools:**

- `confluence_search_tool`  
  - Input: query, tags (e.g., “personal‑loan”, “eligibility‑rules”, “escalation”).  
  - Output: relevant page snippets plus metadata (page_id, version, section headings).  

- `confluence_fetch_tool`  
  - Input: doc_id(s).  
  - Output: full text used in RAG context for the LLM.

- Optional `policy_vector_search_tool`: If you index policy documents into a vector store, you can provide semantic retrieval beyond Confluence search.

**Explainability:** The agent’s structured output should include:

```json
{
  "is_straightforward": true,
  "policy_version": "personal-loan-v2026-W06",
  "applied_rules": [
    {
      "id": "PL-ELIG-001",
      "description": "Min credit score 680 for unsecured loans <= $50k",
      "reference": "Confluence:PersonalLoans#Eligibility"
    },
    {
      "id": "PL-DTI-003",
      "description": "Max DTI 40% for Tier 1 customers",
      "reference": "Confluence:RiskAppetite#DTI"
    }
  ],
  "flags": [
    "NO_MANUAL_REVIEW_REQUIRED"
  ]
}
```

This JSON is later embedded verbatim into the audit record and customer‑facing reasoning (after redaction). [cloud.google](https://cloud.google.com/blog/products/ai-machine-learning/build-a-deep-research-agent-with-google-adk)

***

### 5) RiskScoringAgent

**Role:** Combine the internal ML risk score, historical decision patterns, and current policy output into a numeric and categorical risk assessment.

**Responsibilities:**

- Invoke the internal Risk Scoring model (0–100) with aggregated features from the session state (credit data, fraud, income, employment).  
- Query historical decision outcomes DB for similar profiles (K‑NN like cohort or rules) to produce expected default rate or delta from baseline.  
- Produce a risk summary: `risk_score`, `risk_band`, `expected_loss_estimate`, along with model version for audit.

**Tools:**

- `risk_model_tool`  
  - Input: model_version, feature vector (credit score, DTI, fraud signals, income stability, etc.).  
  - Output: numeric score, explanation features (if model supports SHAP‑like attributions).

- `historical_decisions_tool`  
  - Input: minimal non‑identifying features (banded score ranges, product type, loan size bucket).  
  - Output: aggregated statistics only (e.g., default rate, average loss), preserving PII constraints.

**State:** Write `session.state['risk_assessment']` with both machine score and derived bands.

***

### 6) DecisionAndReportingAgent

**Role:** Turn all upstream signals into a final structured risk report + decision (auto‑approve / auto‑deny / escalate), with a clear reasoning chain and audit‑ready structure.

**Responsibilities:**

- Combine `application_summary`, `credit_data`, `fraud_signals`, `employment_data`, `bank_income_summary`, `policy_verdict`, and `risk_assessment` from state.  
- Apply deterministic business logic on top of LLM reasoning:
  - If `policy_verdict.is_straightforward` is true and fraud low and risk_band ∈ {“Low”, “Medium”} and all required data present → eligible for auto‑decision path (approve or deny).  
  - If any “manual_review_required” flag or risk_band “High” or conflicting signals → escalate.  
- Generate a **structured report**, for example:

```json
{
  "application_id": "12345",
  "decision": "APPROVED",
  "auto_decided": true,
  "loan_terms": {
    "amount": 25000,
    "rate_tier": "B",
    "tenor_months": 36
  },
  "reasons": [
    "Credit score above minimum threshold (Policy PL-ELIG-001).",
    "Debt-to-income ratio within allowed range for Tier 1.",
    "No significant fraud signals detected.",
    "Risk score in 'Medium' band; within acceptable risk appetite."
  ],
  "contra_reasons": [
    "One recent inquiry within last 30 days (non-material)."
  ],
  "policy_references": [ ... ],
  "model_metadata": {
    "risk_model_version": "RM-2026-02",
    "input_features_hash": "abc123..."
  },
  "data_sources": {
    "credit_bureau": "Equifax",
    "fraud_api": "InternalFraudAPI-v3",
    "employment_verification": "ThirdPartyAPI-X"
  },
  "timestamp": "2026-02-09T10:15:00Z"
}
```

**Tools:**

- `persist_audit_record_tool`  
  - Stores the report plus minimal input metadata in an audit store compliant with PCI/PII rules; sensitive fields should be tokenized or omitted; store pointers to secure vault records instead of raw PII.  
- `notify_customer_tool`  
  - Sends decision back to core loan system / customer channel with a customer‑friendly explanation; uses only non‑sensitive explanation strings.

***

### 7) EscalationAgent (Human‑in‑the‑loop)

**Role:** Prepare seamless human escalation and ensure no data re‑entry.

**Responsibilities:**

- Assemble a **human‑readable case file** from the same structured report plus raw tool outputs (where allowed).  
- Add specific questions/prompts for the human underwriter:
  - “Confirm whether recent job change impacts stability.”  
  - “Evaluate borderline DTI in light of high savings buffer.”  
- Assign case to an underwriter queue and await human decision via back‑office UI.

**Tools:**

- `create_case_tool`  
  - Input: report, application_id, reference ids.  
  - Output: `case_id`.  
- `update_case_with_human_decision_tool`  
  - Called from human‑facing system (not LLM) to write human decision back.  
- `finalize_decision_tool`  
  - Applies human decision to core system and logs to audit store.

Because all data is already structured in the ADK session and audit store, the human doesn’t re‑enter data; they simply add the final verdict and comments.

***

## State management and PII/statelessness

### Within‑session state

- ADK provides a per‑Session `state` dictionary accessible across sub‑agents in the MAS, acting as a scratchpad for this **single** application. [codelabs.developers.google](https://codelabs.developers.google.com/codelabs/production-ready-ai-with-gc/3-developing-agents/build-a-multi-agent-system-with-adk)
- Only transient data required for reasoning lives here; PII should be stored as **tokens/handles** referencing an external PCI‑compliant vault or tokenization service.  
- Use templating to inject state into agent prompts (`{credit_data?}`, `{policy_verdict?}`) without persisting that state beyond the session. [google.github](https://google.github.io/adk-docs/sessions/state/)

### Beyond the session (statelessness requirement)

- Credit data must not be stored after the decision session completes:
  - Immediately after final decision, run a cleanup step:  
    - Wipe `session.state` keys with PII or PII‑derived content.  
    - End or delete the session via the SessionService.  
- The **audit store** must be designed to avoid raw credit data:
  - Store: application_id, decision, high‑level metrics (score bands, yes/no flags), policy IDs used, model version, tool call logs (with tool IDs and timestamps).  
  - Omit or tokenize: full credit reports, PANs, SSNs, full addresses, detailed payment histories (store only summary features, e.g., “2 delinquencies in last 24 months”).  
- PCI‑DSS:
  - Keep card/bank identifiers in specialized PCI systems; ADK only sees tokens, last4 where needed for explanation (if allowed by policy), and aggregated derived metrics (e.g., monthly income).  
  - Isolate ADK runtime from systems that store cardholder data; calls to those systems must go through tools that enforce PCI boundaries.

***

## Error handling and resilience

### API‑level error handling

Each external API tool is wrapped with:

- **Retry logic:** One or two retries with exponential backoff for transient failures.  
- **Timeouts:** If employment verification exceeds the 5–30 sec typical SLA, mark that data as “pending” and proceed with partial information, optionally deferring final decision or routing to manual review.  
- **Fallbacks:**
  - Credit Bureau down → rely on bank statement income + internal behavioral data; automatically mark as “manual review required” or use a conservative fallback model.  
  - Fraud API down → tighten thresholds or force manual review.  

Errors are recorded via `log_event_tool` and included in the audit record as “data limitations” annotations.

### Workflow‑level error handling

- Use MAS + LoopAgent patterns for long‑running or multi‑step flows (e.g., polling employment status). If a step fails irrecoverably, write `session.state['status'] = 'ERROR'` and route to EscalationAgent. [github](https://github.com/restatedev/restate-google-adk-example)
- For truly durable orchestration (e.g., hours‑long employer callbacks), consider integrating ADK with an external workflow engine (Restate example, GCP Workflows) that can resume an ADK run with a fresh Session when all data is ready. [github](https://github.com/restatedev/restate-google-adk-example)

***

## Rate limiting and concurrency

- Enforce API rate limits in tool wrappers, not in agent logic:
  - Before each Credit Bureau call, `rate_limit_counter_tool` increments a per‑minute counter; if `>100`, either:
    - Queue the call (delay within acceptable SLA), or  
    - Defer to next minute or manual processing.  
- Use Cloud Armor / API Gateway rate limiting on the upstream front‑door if necessary; but for credit bureau specifically, you’ll usually implement *application‑level* rate limits keyed by API credential rather than IP. [docs.cloud.google](https://docs.cloud.google.com/armor/docs/rate-limiting-overview)
- The Orchestrator can also adjust scheduling:
  - Under heavy load, prioritize “high‑value” or “shorter path” applications, and push borderline cases to manual queues.

***

## Audit trail and explainability

- Every tool invocation should emit a structured audit event:
  - `event_type`, `timestamp`, `agent_name`, `tool_name`, `input_summary`, `output_summary`, `correlation_id`.  
- The final risk report includes:
  - Policy rules IDs and document references (Confluence URLs/section IDs).  
  - Risk model version and high‑level feature contributions.  
  - Simple natural‑language reasons and contra‑reasons suitable for regulators and customers.  
- ADK’s shared state makes it easy to embed a “stage” variable that indicates which step produced which part of the reasoning chain. [cloud.google](https://cloud.google.com/blog/products/ai-machine-learning/build-a-deep-research-agent-with-google-adk)
- Keep a clear mapping from every final conclusion back to either:
  - A specific policy rule,  
  - A model output,  
  - A human underwriter judgment.

***

## Putting it together (flow)

1. Application arrives → OrchestratorAgent creates session, populates control state, logs “start”.  
2. DataCollectionAgent validates and normalizes input; if OK, marks `is_straightforward_candidate = true`.  
3. CreditAndFraudAgent runs external APIs in parallel, respecting rate limits and capturing partial failures.  
4. PolicyReasoningAgent fetches latest policy docs and outputs `policy_verdict` with rule IDs and policy version.  
5. RiskScoringAgent calls the internal risk model and historical DB, writes `risk_assessment`.  
6. DecisionAndReportingAgent combines everything:
   - If straightforward and low risk → auto‑approve/deny within minutes.  
   - Else → EscalationAgent prepares case for human review.  
7. Decision is persisted (without PII), customer is notified, session state is wiped, and session is closed.

This architecture satisfies: parallel API orchestration, dynamic policy ingestion, explainable structured reports, regulatory auditability, per‑application isolation, stateless PII handling, graceful error behavior, and adherence to external rate limits, all using ADK’s multi‑agent and session‑state patterns. [google.github](https://google.github.io/adk-docs/agents/multi-agents/)