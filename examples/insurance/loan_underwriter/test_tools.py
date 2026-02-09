#!/usr/bin/env python3
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src import LoanApplication
from src.tools import (
    fetch_credit_bureau, fetch_employment_verification,
    fetch_fraud_signals, fetch_bank_statement_analysis,
    evaluate_policy, call_risk_model, build_risk_report
)

SAMPLE_APP = {
    "application_id": "test-001",
    "customer_id": "test-cust-001",
    "name": "John Doe",
    "ssn_last4": "1234",
    "income": 85000.0,
    "employer": "Tech Corp",
    "loan_amount": 25000.0,
    "purpose": "Home improvement",
}

def test_single_tools():
    print("🧪 Testing individual tools...")
    
    # Credit bureau (should respect rate limits)
    credit = fetch_credit_bureau(SAMPLE_APP)
    print(f"   Credit: {credit.get('score', credit)}")
    
    # Fraud (fast)
    fraud = fetch_fraud_signals(SAMPLE_APP)
    print(f"   Fraud score: {fraud['fraud_score']}")
    
    # Policy evaluation
    policy_result = evaluate_policy({"credit": credit, "fraud": fraud, "application": SAMPLE_APP})
    print(f"   Policy decision: {policy_result}")

def test_full_pipeline():
    print("\n🔄 Testing FULL pipeline (parallel simulation)...")
    start_time = time.time()
    
    # Simulate parallel calls (in reality, ADK agent does this)
    credit = fetch_credit_bureau(SAMPLE_APP)
    fraud = fetch_fraud_signals(SAMPLE_APP)
    employment = fetch_employment_verification(SAMPLE_APP)
    bank = fetch_bank_statement_analysis(SAMPLE_APP)
    
    policy = evaluate_policy({
        "application": SAMPLE_APP,
        "credit": credit,
        "fraud": fraud,
        "employment": employment,
        "bank": bank,
    })
    
    risk = call_risk_model({
        "application": SAMPLE_APP,
        "credit": credit,
        "fraud": fraud,
    })
    
    final_report = build_risk_report({
        "application": SAMPLE_APP,
        "credit": credit,
        "fraud": fraud,
        "employment": employment,
        "bank": bank,
        "risk_score": risk,
        "policy_decision": policy,
        "session_id": "test-sess-001",
    })
    
    elapsed = time.time() - start_time
    print(f"✅ Pipeline complete in {elapsed:.1f}s")
    print(f"   Decision: {final_report['decision']['decision']}")
    print(f"   Straight-through: {final_report['decision']['is_straight_through']}")
    print(f"   Risk score: {final_report.get('risk_score', {}).get('score', 'N/A')}")

def test_rate_limiting():
    print("\n⏱️  Testing rate limiting...")
    start = time.time()
    for i in range(110):  # Over limit
        result = fetch_credit_bureau({**SAMPLE_APP, "application_id": f"rate-test-{i}"})
        if "rate_limited" in str(result):
            print(f"   Rate limit hit at call #{i+1}")
            break
    print(f"   Rate test complete in {time.time() - start:.1f}s")

if __name__ == "__main__":
    test_single_tools()
    test_full_pipeline()
    test_rate_limiting()
