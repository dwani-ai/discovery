Scenario 1: Financial Services - Agent Development (ADK)

Problem Statement

A fintech company provides personal loans ranging from $5,000 to $100,000. They currently process 10,000 loan applications daily, with an average approval time of 48 hours. This delay is causing customer churn—competitors are offering same-day approvals.


Current workflow:

Customer submits application via web/mobile (basic info: income, employer, loan amount)

Operations team manually queries credit bureaus (Equifax, TransUnion, Experian)

Employment verification team calls employers or uses third-party verification APIs

Underwriters cross-reference internal policy documents to determine eligibility

Senior underwriter reviews borderline cases

Decision communicated to customer

The bottleneck:

Policy documents change weekly (new regulations, risk appetite adjustments)

Underwriters spend 30 minutes per application navigating 200+ pages of policy PDFs

The internal policy engine is a legacy rule system that can't keep pace with policy changes

70% of applications are 'straightforward' (clear approve/deny based on policies) but still take 48 hours

External APIs available:

Credit Bureau API: Returns credit score, payment history, outstanding debts, inquiries (latency: 2-3 seconds)

Employment Verification API: Confirms current employment, salary range, tenure (latency: 5-30 seconds, may require async callback)

Fraud Detection API: Real-time fraud signals, identity verification (latency: 1 second)

Bank Statement Analysis API: Parses uploaded statements for income verification (latency: 10-15 seconds)

Internal resources:

Policy documents in Confluence (200+ pages, updated weekly)

Historical decisions database (5 years of approve/deny decisions with outcomes)

Risk scoring model (custom ML model returning a score 0-100)

The customer wants: An agentic system using Google's Agent Development Kit that can:

Orchestrate calls to multiple external APIs in parallel where possible

Apply current business rules from policy documents (these change frequently)

Generate a structured risk report with clear reasoning chain

Auto-approve/deny 70% of 'straightforward' applications in under 5 minutes

Route complex cases to human underwriters with pre-populated analysis

Key requirements:

Must maintain complete audit trail for regulatory compliance

Each application must be isolated—no data leakage between sessions

System must handle API failures gracefully (credit bureau downtime is common)

Reasoning must be explainable—regulators may ask why a decision was made

Must respect rate limits on external APIs (credit bureau: 100 calls/minute)

Constraints:

Cannot store credit data beyond the session—must be stateless for PII compliance

Must support PCI-DSS compliance for any financial data handling

Human escalation path must be seamless (no re-entry of data)


Design an agent architecture using ADK. Sketch out the sub-agents, their responsibilities, tool definitions, and how you would handle state management and error recovery.