import os, time, requests
from typing import Dict, Any
from .models import (
    LoanApplication, CreditReport, EmploymentVerification,
    FraudSignals, BankAnalysis, RiskScore, Decision, RiskReport, ReasoningStep
)

CREDIT_BUREAU_BASE = os.getenv("CREDIT_BUREAU_URL")
EMPLOYMENT_API_BASE = os.getenv("EMPLOYMENT_API_URL")
FRAUD_API_BASE = os.getenv("FRAUD_API_URL")
BANK_API_BASE = os.getenv("BANK_API_URL")
POLICY_SERVICE_URL = os.getenv("POLICY_SERVICE_URL")
RISK_MODEL_URL = os.getenv("RISK_MODEL_URL")

# ---- External API tools ----

def fetch_credit_bureau(application: Dict[str, Any]) -> Dict[str, Any]:
    """Call credit bureau API (respecting rate limits via external limiter)."""
    app = LoanApplication(**application)
    resp = requests.post(
        f"{CREDIT_BUREAU_BASE}/score",
        json={"ssn_last4": app.ssn_last4, "name": app.name},
        timeout=5,
    )
    resp.raise_for_status()
    data = resp.json()
    credit = CreditReport(
        bureau=data["bureau"],
        score=data["score"],
        delinquencies=data["delinquencies"],
        utilization=data["utilization"],
        inquiries_12m=data["inquiries_12m"],
        trade_lines=data["trade_lines"],
    )
    return credit.model_dump()

def fetch_employment_verification(application: Dict[str, Any]) -> Dict[str, Any]:
    app = LoanApplication(**application)
    resp = requests.post(
        f"{EMPLOYMENT_API_BASE}/verify",
        json={"employer": app.employer, "name": app.name},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    ev = EmploymentVerification(
        employer=data["employer"],
        status=data["status"],
        salary_range=data["salary_range"],
        tenure_months=data["tenure_months"],
    )
    return ev.model_dump()

def fetch_fraud_signals(application: Dict[str, Any]) -> Dict[str, Any]:
    app = LoanApplication(**application)
    resp = requests.post(
        f"{FRAUD_API_BASE}/signals",
        json={"customer_id": app.customer_id, "loan_amount": app.loan_amount},
        timeout=3,
    )
    resp.raise_for_status()
    data = resp.json()
    fraud = FraudSignals(
        fraud_score=data["fraud_score"],
        flags=data.get("flags", []),
    )
    return fraud.model_dump()

def fetch_bank_statement_analysis(application: Dict[str, Any]) -> Dict[str, Any]:
    app = LoanApplication(**application)
    resp = requests.post(
        f"{BANK_API_BASE}/analyze",
        json={"customer_id": app.customer_id},
        timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    bank = BankAnalysis(
        avg_monthly_income=data["avg_monthly_income"],
        avg_monthly_expenses=data["avg_monthly_expenses"],
        nsf_events_12m=data["nsf_events_12m"],
    )
    return bank.model_dump()

# ---- Policy and risk tools ----

def fetch_current_policy() -> Dict[str, Any]:
    """Fetch versioned policy from your policy service (backed by Confluence)."""
    resp = requests.get(POLICY_SERVICE_URL, timeout=5)
    resp.raise_for_status()
    return resp.json()  # include "version" and structured rules

def evaluate_policy(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic rule engine over normalized inputs.
    Returns decision + rule hits.
    """
    policy = fetch_current_policy()
    rules = policy["rules"]
    hits = []
    decision = "manual_review"

    credit = inputs.get("credit", {})
    fraud = inputs.get("fraud", {})

    if credit.get("score", 0) >= rules["min_score_for_auto_approve"] \
       and fraud.get("fraud_score", 1.0) <= rules["max_fraud_score_for_auto_approve"]:
        decision = "approve"
        hits.append("RULE_AUTO_APPROVE_SCORE_AND_FRAUD")
    elif credit.get("score", 0) < rules["min_score_for_auto_deny"]:
        decision = "deny"
        hits.append("RULE_AUTO_DENY_LOW_SCORE")
    else:
        decision = "manual_review"
        hits.append("RULE_COMPLEX_CASE")

    return {
        "decision": decision,
        "policy_version": policy["version"],
        "rule_hits": hits,
    }

def call_risk_model(features: Dict[str, Any]) -> Dict[str, Any]:
    """Call internal ML model to get risk score."""
    resp = requests.post(RISK_MODEL_URL, json=features, timeout=5)
    resp.raise_for_status()
    data = resp.json()
    rs = RiskScore(score=data["score"], model_version=data["model_version"])
    return rs.model_dump()

def build_risk_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble a structured, explainable risk report."""
    app = LoanApplication(**payload["application"])
    credit = CreditReport(**payload["credit"]) if payload.get("credit") else None
    employment = EmploymentVerification(**payload["employment"]) if payload.get("employment") else None
    fraud = FraudSignals(**payload["fraud"]) if payload.get("fraud") else None
    bank = BankAnalysis(**payload["bank"]) if payload.get("bank") else None
    risk = RiskScore(**payload["risk_score"]) if payload.get("risk_score") else None
    decision_raw = payload["policy_decision"]

    decision = Decision(
        decision=decision_raw["decision"],
        reason_summary="; ".join(decision_raw["rule_hits"]),
        policy_version=decision_raw["policy_version"],
        is_straight_through=decision_raw["decision"] in ["approve", "deny"],
    )

    reasoning = [
        ReasoningStep(
            step="credit_policy",
            details=f"Credit score {credit.score} triggered {decision_raw['rule_hits']}",
            evidence={"score": str(credit.score)} if credit else {},
        ),
        ReasoningStep(
            step="fraud_check",
            details=f"Fraud score {fraud.fraud_score}" if fraud else "Fraud data unavailable",
            evidence={"fraud_score": str(fraud.fraud_score)} if fraud else {},
        ),
    ]

    report = RiskReport(
        application=app,
        credit=credit,
        employment=employment,
        fraud=fraud,
        bank=bank,
        risk_score=risk,
        decision=decision,
        reasoning_chain=reasoning,
        artifacts={},
    )
    return report.model_dump()
