# src/agent.py
"""Loan Underwriting ADK Agents."""

from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from .tools import (
    fetch_credit_bureau, fetch_employment_verification,
    fetch_fraud_signals, fetch_bank_statement_analysis,
    call_risk_model, evaluate_policy, build_risk_report
)

import os
API_BASE_URL = os.environ.get("DWANI_API_BASE_URL", "http://localhost:8000/v1")
MODEL_NAME = os.environ.get("DWANI_MODEL_NAME", "qwen3-coder")

# LiteLLM connector for ADK agents
# Note: If your proxy requires a key, set it via env var expected by your proxy/LiteLLM.
model = LiteLlm(
    model=MODEL_NAME,
    api_base=API_BASE_URL,
    provider="openai"
)

# Data ingestion agent
data_agent = LlmAgent(
    model=model,
    name="data_ingestion_agent",
    instruction=(
        "Fetch ALL external loan data:\n"
        "• fetch_credit_bureau\n"
        "• fetch_employment_verification\n"
        "• fetch_fraud_signals\n"
        "• fetch_bank_statement_analysis\n"
        "Return JSON with all results."
    ),
    tools=[
        fetch_credit_bureau,
        fetch_employment_verification,
        fetch_fraud_signals,
        fetch_bank_statement_analysis,
    ],
)

# Risk + policy agent
risk_agent = LlmAgent(
    model=model,
    name="risk_policy_agent", 
    instruction=(
        "Compute risk score and apply policy:\n"
        "• call_risk_model\n"
        "• evaluate_policy\n"
        "Return: {\"risk_score\": ..., \"policy_decision\": ...}"
    ),
    tools=[call_risk_model, evaluate_policy],
)

# Report agent
report_agent = LlmAgent(
    model=model,
    name="report_agent",
    instruction="Call build_risk_report → return RiskReport JSON exactly.",
    tools=[build_risk_report],
)

# ROOT AGENT (ADK entrypoint)
root_agent = LlmAgent(
    model=model,
    name="loan_orchestrator",
    instruction=(
        "Loan underwriting orchestrator:\n"
        "1. data_ingestion_agent → fetch data\n" 
        "2. risk_policy_agent → analyze\n"
        "3. report_agent → RiskReport JSON\n\n"
        "Input: {\"application\": LoanApplication}\n"
        "Output: RiskReport JSON"
    ),
    sub_agents=[data_agent, risk_agent, report_agent],
)
