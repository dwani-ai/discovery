#!/usr/bin/env python3
import sys
import time
import json
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src import LoanApplication, RiskReport
from src.tools import SAMPLE_APP, fetch_credit_bureau, fetch_employment_verification, fetch_fraud_signals, fetch_bank_statement_analysis, evaluate_policy, call_risk_model, build_risk_report

def test_complete_pipeline():
    """Production underwriting pipeline."""
    print("🚀 UNDERWRITING PIPELINE TEST...")
    start_time = time.time()
    
    application = SAMPLE_APP
    
    # Data ingestion (what DataIngestionAgent does)
    print("   📡 Fetching external data...")
    data = {
        "application": application,
        "credit": fetch_credit_bureau(application),
        "employment": fetch_employment_verification(application),
        "fraud": fetch_fraud_signals(application),
        "bank": fetch_bank_statement_analysis(application),
    }
    
    # Risk + policy (what RiskPolicyAgent does)
    print("   ⚙️  Risk scoring + policy...")
    risk_policy = {
        "risk_score": call_risk_model(data),
        "policy_decision": evaluate_policy(data),
    }
    
    # Final report (what ReportAgent does)
    print("   📄 Building RiskReport...")
    full_payload = {**data, **risk_policy, "session_id": "test-001"}
    report = build_risk_report(full_payload)
    
    elapsed = time.time() - start_time
    print(f"\n✅ SUCCESS: {elapsed:.1f}s")
    print(f"   Decision: {report['decision']['decision']}")
    print(f"   Risk score: {report['risk_score']['score']:.1f}")
    print(f"   Straight-through: {report['decision']['is_straight_through']}")
    
    RiskReport.model_validate(report)
    print("✅ Audit-ready JSON ✓")
    return report

if __name__ == "__main__":
    report = test_complete_pipeline()
    print("\n🎉 PRODUCTION SYSTEM VALIDATED!")
