```ascii
                             ┌───────────────────────────┐
                             │       Channels & LOS      │
                             │───────────────────────────│
   ┌────────────────────┐    │  Web / Mobile Apps        │
   │                    │    │  Call Center / Branch UI  │
   │   Customers        │───▶│  Loan Origination System  │
   │                    │    └─────────────┬─────────────┘
   └────────────────────┘                  │
                                           │ Loan application
                                           ▼
                          ┌──────────────────────────────────────┐
                          │      ADK Loan Decisioning Service    │
                          │──────────────────────────────────────│
                          │  API Layer                          │
                          │  ─────────                          │
                          │  - Loan Decisioning API (REST/gRPC) │
                          │  - AuthN/AuthZ, Validation          │
                          ├──────────────────────────────────────┤
                          │      ADK Runtime (Multi-Agent)      │
                          │                                      │
                          │   ┌───────────────────────────────┐  │
                          │   │       OrchestratorAgent      │  │
                          │   └──────────────┬───────────────┘  │
                          │                  │ plans / routes    │
                          │   ┌──────────────┴────────────────┐  │
                          │   │          Sub-Agents           │  │
                          │   │───────────────────────────────│  │
                          │   │  DataCollectionAgent          │  │
                          │   │  CreditAndFraudAgent          │  │
                          │   │  PolicyReasoningAgent         │  │
                          │   │  RiskScoringAgent             │  │
                          │   │  DecisionAndReportingAgent    │  │
                          │   │  EscalationAgent (human-in-loop)││
                          │   └───────────────────────────────┘  │
                          ├──────────────────────────────────────┤
                          │           Session & State            │
                          │           ───────────────            │
                          │  - SessionService (DB, non-PII)      │
                          │  - Per-application state:            │
                          │    tokens + derived features only    │
                          ├──────────────────────────────────────┤
                          │              Tools Layer             │
                          │              ───────────             │
                          │  - Audit Logging Tool                │
                          │  - Rate Limiting Tool                │
                          │  - Policy / Confluence Tools         │
                          │  - Risk Model & History Tools        │
                          │  - Case / Notification Tools         │
                          └───────────────┬──────────────────────┘
                                          │ tool calls
                                          ▼
   ┌──────────────────────────────────────────────────────────────────────┐
   │          External Systems & Internal Services                        │
   │──────────────────────────────────────────────────────────────────────│
   │  Risk & Data APIs:                                                   │
   │   - Credit Bureau API             - Fraud Detection API              │
   │   - Employment Verification API   - Bank Statement Analysis API      │
   │                                                                       
   │  Internal Services:                                                   │
   │   - Risk Scoring Model Service    - Historical Decisions Database    │
   │   - Policy Repository (Confluence + index)                           │
   │   - Case Management / Underwriter Workbench                          │
   │   - Tokenization / PII Vault (PCI zone, tokens only to ADK)          │
   └──────────────────────────────────────────────────────────────────────┘

      ▲                                           ▲
      │ audit logs, metrics, traces              │ tokens / features
      │                                           │
   ┌──┴───────────────────────────────┐
   │ Security, Governance, Observability │
   │────────────────────────────────────│
   │ - WAF / API Gateway, IAM, network  │
   │ - PCI-DSS controls, PII vault      │
   │ - Append-only audit log            │
   │ - Metrics, traces, drift alerts    │
   └────────────────────────────────────┘
```