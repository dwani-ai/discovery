#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from src import LoanApplication, RiskReport, Decision

def test_models():
    app = LoanApplication(
        application_id="app-123",
        customer_id="cust-456",
        name="John Doe",
        ssn_last4="1234",
        income=85000.0,
        employer="Tech Corp",
        loan_amount=25000.0,
        purpose="Home improvement",
    )
    print("✅ LoanApplication:", app.model_dump_json(indent=2))

    report = RiskReport(
        application=app,
        session_id="sess-789",
        decision=Decision(
            decision="approve",
            reason_summary="High credit score, verified employment",
            policy_version="v2026-02-09",
            is_straight_through=True,
        ),
    )
    print("✅ RiskReport:", report.model_dump_json(indent=2))

if __name__ == "__main__":
    test_models()
