import os, time, threading, random, requests
from typing import Dict, Any, List
from datetime import datetime
from dotenv import load_dotenv
from .models import (
    LoanApplication, CreditReport, EmploymentVerification,
    FraudSignals, BankAnalysis, RiskScore
)

load_dotenv()

# Global rate limiter (token bucket for credit bureau: 100/min)
CREDIT_BUCKET_CAPACITY = 100
CREDIT_REFILL_INTERVAL = 60  # seconds
_credit_tokens = CREDIT_BUCKET_CAPACITY
_credit_lock = threading.Lock()
_credit_last_refill = time.time()

def _acquire_credit_token() -> bool:
    """Token bucket rate limiter for credit bureau (100 calls/min)."""
    global _credit_tokens, _credit_last_refill
    with _credit_lock:
        now = time.time()
        elapsed = now - _credit_last_refill
        refill_amount = (elapsed / CREDIT_REFILL_INTERVAL) * CREDIT_BUCKET_CAPACITY
        _credit_tokens = min(CREDIT_BUCKET_CAPACITY, _credit_tokens + refill_amount)
        _credit_last_refill = now
        
        if _credit_tokens <= 0:
            return False
        _credit_tokens -= 1
        return True

# ---- MOCK EXTERNAL API TOOLS (replace with real URLs later) ----

def fetch_credit_bureau(application: Dict[str, Any]) -> Dict[str, Any]:
    """
    Credit Bureau API: 2-3s latency, rate limited, 10% failure rate.
    """
    if not _acquire_credit_token():
        return {"error": "rate_limited", "unavailable": True}
    
    app = LoanApplication(**application)
    # Simulate 2-3s latency + 10% failure
    time.sleep(random.uniform(2.0, 3.0))
    if random.random() < 0.1:  # 10% failure
        return {"error": "bureau_unavailable", "unavailable": True}
    
    bureau = random.choice(["Equifax", "TransUnion", "Experian"])
    score = random.randint(650 if app.income > 50000 else 550, 800)
    
    return CreditReport(
        bureau=bureau,
        score=score,
        delinquencies=random.randint(0, 2),
        utilization=round(random.uniform(0.1, 0.4), 2),
        inquiries_12m=random.randint(0, 6),
        trade_lines=random.randint(5, 15),
    ).model_dump()

def fetch_employment_verification(application: Dict[str, Any]) -> Dict[str, Any]:
    """Employment API: 5-30s latency, 5% async callback needed."""
    app = LoanApplication(**application)
    time.sleep(random.uniform(5.0, 30.0))  # High variance
    
    if random.random() < 0.05:  # 5% needs callback
        return {"status": "async_pending", "employer": app.employer}
    
    status = "verified" if random.random() < 0.9 else "mismatch"
    return EmploymentVerification(
        employer=app.employer,
        status=status,
        salary_range=f"${int(app.income*0.8):,}–${int(app.income*1.2):,}",
        tenure_months=random.randint(12, 120),
    ).model_dump()

def fetch_fraud_signals(application: Dict[str, Any]) -> Dict[str, Any]:
    """Fraud API: 1s latency, real-time."""
    time.sleep(1.0)
    app = LoanApplication(**application)
    fraud_score = random.uniform(0.01, 0.3)  # Usually low
    flags = []
    if fraud_score > 0.2:
        flags = ["high_velocity_applications", "recent_ip_change"]
    
    return FraudSignals(fraud_score=fraud_score, flags=flags).model_dump()

def fetch_bank_statement_analysis(application: Dict[str, Any]) -> Dict[str, Any]:
    """Bank API: 10-15s latency."""
    time.sleep(random.uniform(10.0, 15.0))
    app = LoanApplication(**application)
    income_variance = random.uniform(0.85, 1.15)
    return BankAnalysis(
        avg_monthly_income=app.income / 12 * income_variance,
        avg_monthly_expenses=(app.income / 12) * random.uniform(0.6, 0.9),
        nsf_events_12m=random.randint(0, 3),
    ).model_dump()

# ---- INTERNAL TOOLS ----

def fetch_current_policy() -> Dict[str, Any]:
    """Fetch latest policy rules (weekly updates)."""
    time.sleep(0.5)  # Policy service latency
    return {
        "version": f"v2026-W{f'{datetime.now().isocalendar()[1]:02d}'}",
        "rules": {
            "min_score_for_auto_approve": 720,
            "max_fraud_score_for_auto_approve": 0.15,
            "min_score_for_auto_deny": 580,
            "max_dti_for_approve": 0.36,
            "max_loan_income_ratio": 0.4,
        }
    }

def evaluate_policy(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deterministic policy engine. Handles missing data gracefully.
    """
    policy = fetch_current_policy()
    rules = policy["rules"]
    credit = inputs.get("credit", {})
    fraud = inputs.get("fraud", {})
    app = inputs.get("application", {})
    
    hits = []
    decision = "manual_review"
    
    # Graceful degradation for missing data
    if credit.get("unavailable") or fraud.get("unavailable"):
        hits.append("RULE_DATA_UNAVAILABLE")
        return {"decision": "manual_review", "policy_version": policy["version"], "rule_hits": hits}
    
    score = credit.get("score", 0)
    fraud_score = fraud.get("fraud_score", 1.0)
    
    # Straight-through processing logic (70% target)
    if score >= rules["min_score_for_auto_approve"] and fraud_score <= rules["max_fraud_score_for_auto_approve"]:
        decision = "approve"
        hits.append("RULE_AUTO_APPROVE_HIGH_SCORE_LOW_FRAUD")
    elif score < rules["min_score_for_auto_deny"]:
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
    """Internal ML risk scoring model."""
    time.sleep(2.0)  # Model inference
    # Simplified risk score calculation
    credit_weight = features.get("credit", {}).get("score", 500) / 850
    fraud_weight = 1.0 - features.get("fraud", {}).get("fraud_score", 1.0)
    income_weight = min(1.0, features.get("application", {}).get("income", 0) / 100000)
    
    score = (credit_weight * 0.5 + fraud_weight * 0.3 + income_weight * 0.2) * 100
    return RiskScore(score=round(score, 2), model_version="risk-v2.1").model_dump()

def build_risk_report(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Assemble final audit-ready report."""
    from .models import RiskReport, ReasoningStep, Decision
    
    app = LoanApplication(**payload["application"])
    credit = CreditReport(**payload["credit"]) if payload.get("credit") else None
    employment = EmploymentVerification(**payload["employment"]) if payload.get("employment") else None
    fraud = FraudSignals(**payload["fraud"]) if payload.get("fraud") else None
    bank = BankAnalysis(**payload["bank"]) if payload.get("bank") else None
    risk = RiskScore(**payload["risk_score"]) if payload.get("risk_score") else None
    policy_decision = payload["policy_decision"]
    
    decision = Decision(
        decision=policy_decision["decision"],
        reason_summary="; ".join(policy_decision["rule_hits"]),
        policy_version=policy_decision["policy_version"],
        is_straight_through=policy_decision["decision"] in ["approve", "deny"],
    )
    
    reasoning = []
    if credit:
        reasoning.append(ReasoningStep(
            step="credit_assessment",
            details=f"Score {credit.score} from {credit.bureau}",
            evidence={"score": str(credit.score), "bureau": credit.bureau},
        ))
    if fraud:
        reasoning.append(ReasoningStep(
            step="fraud_check",
            details=f"Fraud score {fraud.fraud_score}",
            evidence={"fraud_score": str(fraud.fraud_score)},
        ))
    
    report = RiskReport(
        application=app,
        credit=credit,
        employment=employment,
        fraud=fraud,
        bank=bank,
        risk_score=risk,
        decision=decision,
        reasoning_chain=reasoning,
        session_id=payload.get("session_id", "unknown"),
    )
    return report.model_dump()
