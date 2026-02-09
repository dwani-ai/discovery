```ascii
   Customer       Web/Mobile      LOS          PII Vault        ADK API        Orchestrator       Sub-Agents & Tools                 External & Internal
      |               |             |               |              |                 |                     |                                |
      | 1) Fill app   |             |               |              |                 |                     |                                |
      |──────────────▶|             |               |              |                 |                     |                                |
      |               |             |               |              |                 |                     |                                |
      |               | 2) Submit app + docs        |              |                 |                     |                                |
      |               |────────────────────────────▶|              |                 |                     |                                |
      |               |             |               |              |                 |                     |                                |
      |               |             | 3) Store raw PII, get tokens |                 |                     |                                |
      |               |             |─────────────────────────────▶|                 |                     |                                |
      |               |             |             pii_token/doc_tokens             |                     |                                |
      |               |             |◀─────────────────────────────|                 |                     |                                |
      |               |             |               |              |                 |                     |                                |
      |               |             | 4) Loan app (tokens + derived features, no raw PII)                 |                                |
      |               |             |──────────────────────────────────────────────▶|                     |                                |
      |               |             |               |              |                 |                     |                                |
      |               |             |               | 5) Auth, validate, create Session                    |                                |
      |               |             |               |              |────────────────▶|                     |                                |
      |               |             |               |              | 6) Start decision (session handle)    |                                |
      |               |             |               |              |──────────────────────────────────────▶|                                |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 | 7) DataCollectionAgent: validate/normalize           |
      |               |             |               |              |                 |────────────────────────────────────▶ DataCollAgent   |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 |◀──────────────────────────────────── Normalized data |
      |               |             |               |              |                 |  (summary, flags → session.state)                    |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 | 9) CreditAndFraudAgent (parallel external data)      |
      |               |             |               |              |                 |────────────────────────────────────▶ Credit/FraudAg  |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 |                     | 10) Tools → APIs (credit,fraud,|
      |               |             |               |              |                 |                     |     employment, bank stmts)   |
      |               |             |               |              |                 |                     |──────────────────────────────▶|
      |               |             |               |              |                 |                     |    scores, signals, income    |
      |               |             |               |              |                 |                     |◀──────────────────────────────|
      |               |             |               |              |                 |◀──────────────────────────────────── Aggregated data |
      |               |             |               |              |                 |  (credit_data, fraud_signals, income → state)       |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 | 13) PolicyReasoningAgent                             |
      |               |             |               |              |                 |────────────────────────────────────▶ PolicyAgent     |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 |                     | 14) Query Policy Repo          |
      |               |             |               |              |                 |                     |──────────────────────────────▶|
      |               |             |               |              |                 |                     |  Policy docs + metadata       |
      |               |             |               |              |                 |                     |◀──────────────────────────────|
      |               |             |               |              |                 |◀──────────────────────────────────── policy_verdict  |
      |               |             |               |              |                 |   (is_straightforward, rule_ids → state)            |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 | 17) RiskScoringAgent                                 |
      |               |             |               |              |                 |────────────────────────────────────▶ RiskAgent       |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 |                     | 18) Call Risk Model + History  |
      |               |             |               |              |                 |                     |──────────────────────────────▶|
      |               |             |               |              |                 |                     |  risk_score, band, loss       |
      |               |             |               |              |                 |                     |◀──────────────────────────────|
      |               |             |               |              |                 |◀──────────────────────────────────── risk_assessment |
      |               |             |               |              |                 |   (score, band → state)                              |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              |                 | 20) Decision & report (inside Orchestrator or       |
      |               |             |               |              |                 |     DecisionAndReportingAgent)                      |
      |               |             |               |              |                 |   - Apply rules: auto APPROVE/DENY vs ESCALATE      |
      |               |             |               |              |                 |   - Build structured risk_report                     |
      |               |             |               |              |                 |   - Audit log, write report to store                 |
      |               |             |               |              |                 |                     |                                |
      |               |             |               |              | 21) Final decision + risk_report                       |
      |               |             |               |              |◀─────────────────────────────────────── Orchestrator   |
      |               |             |               |              |                 |                     |                                |
      |               |             |               | 22) Decision (APPROVE/DENY/ESCALATE) + explanation                        |
      |               |             |◀─────────────────────────────────────────────── ADK API                                |
      |               |             |               |              |                 |                     |                                |
      |               | 23) Update status, show decision to customer                  |                     |                                |
      |               |◀──────────────────────────── LOS                               |                     |                                |
      | 24) Decision & explanation                                                   |                     |                                |
      |◀──────────────────────────── Web/Mobile App                                  |                     |                                |
      |               |             |               |              |                 |                     |                                |
```