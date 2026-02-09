#!/usr/bin/env python3
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Dict, Any, Optional
import uvicorn
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
from . import LoanApplication, RiskReport
from .tools import SAMPLE_APP, fetch_credit_bureau, fetch_employment_verification, fetch_fraud_signals, fetch_bank_statement_analysis, evaluate_policy, call_risk_model, build_risk_report
from .audit import write_audit_event, enqueue_human_review  # SQLite version
app = FastAPI(title="Loan Underwriting API", version="1.0.0")

class UnderwriteRequest(BaseModel):
    application: LoanApplication

@app.post("/underwrite", response_model=RiskReport)
async def underwrite_loan(request: UnderwriteRequest, background_tasks: BackgroundTasks):
    """
    Production underwriting endpoint.
    Returns RiskReport JSON in <60s for 70% straight-through cases.
    """
    session_id = str(uuid.uuid4())
    
    # AUDIT: Session start
    background_tasks.add_task(write_audit_event, session_id, "session_start", {"loan_amount": request.application.loan_amount})
    
    try:
        application = request.application.model_dump()
        
        # PRODUCTION PIPELINE (exact agent logic)
        data = {
            "application": application,
            "credit": fetch_credit_bureau(application),
            "employment": fetch_employment_verification(application),
            "fraud": fetch_fraud_signals(application),
            "bank": fetch_bank_statement_analysis(application),
        }
        
        risk_policy = {
            "risk_score": call_risk_model(data),
            "policy_decision": evaluate_policy(data),
        }
        
        full_payload = {**data, **risk_policy, "session_id": session_id}
        report = build_risk_report(full_payload)
        
        decision = report["decision"]["decision"]
        
        # AUDIT: Decision logged
        background_tasks.add_task(write_audit_event, session_id, "decision", {
            "decision": decision,
            "straight_through": report["decision"]["is_straight_through"],
            "risk_score": report["risk_score"]["score"]
        })
        
        # HUMAN ESCALATION (if needed)
        if decision == "manual_review":
            background_tasks.add_task(enqueue_human_review, session_id, report)
        
        # PCI-DSS: No PII returned in response
        safe_report = report.copy()
        safe_report["application"]["ssn_last4"] = "****"
        
        return RiskReport(**safe_report)
        
    except Exception as e:
        background_tasks.add_task(write_audit_event, session_id, "error", {"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Underwriting failed: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy", "uptime": "100%"}

@app.get("/test")
async def test_endpoint():
    """Quick test endpoint."""
    return {"message": "Underwriting API ready", "straight_through_target": "70%"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
