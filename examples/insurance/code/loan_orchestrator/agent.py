from google.adk.agents.llm_agent import LlmAgent
from google.adk.tools import FunctionTool
from google.adk.models.lite_llm import LiteLlm
from pydantic import BaseModel, Field
from typing import Any, Dict
import os 


api_base_url = os.environ.get("DWANI_API_BASE_URL", "http://localhost:8000/v1")


model_name_at_endpoint = "qwen3-coder" 


model=LiteLlm(
        model=model_name_at_endpoint,
        api_base=api_base_url,
        )

from google.adk.agents import LlmAgent, Agent
from .tools import (
    fetch_credit_bureau,
    fetch_employment_verification,
    fetch_fraud_signals,
    fetch_bank_statement_analysis,
    evaluate_policy,
    call_risk_model,
    build_risk_report,
)
from .models import RiskReport

# 1) Data Ingestion & Validation Agent
diva_agent = LlmAgent(
    name="data_ingestion_agent",
    model=model,
    description="Fetches external data for a loan application.",
    instruction=(
        "Given a loan application JSON, call the appropriate tools to fetch "
        "credit, employment, fraud, and bank statement data in parallel where possible. "
        "Return a JSON payload with these sections filled."
    ),
    tools=[
        fetch_credit_bureau,
        fetch_employment_verification,
        fetch_fraud_signals,
        fetch_bank_statement_analysis,
    ],
)

# 2) Risk & Policy Agent
risk_policy_agent = LlmAgent(
    name="risk_policy_agent",
    model=model,
    description="Applies policy and risk scoring to normalized data.",
    instruction=(
        "Given normalized data (application + external data), call the risk model "
        "and policy evaluation tools. Return policy_decision, risk_score, and "
        "rule_hits suitable for report building."
    ),
    tools=[evaluate_policy, call_risk_model],
)

# 3) Report Agent (for explainability + pre-population)
report_agent = LlmAgent(
    name="report_agent",
    model=model,
    description="Builds structured risk reports.",
    instruction=(
        "Given application, external data, policy decision, and risk score, "
        "call the build_risk_report tool to produce a final RiskReport JSON."
    ),
    tools=[build_risk_report],
    output_model=RiskReport,  # structured output
)

# 4) Root Orchestrator Agent
root_agent = Agent(
    name="loan_orchestrator",
    model=model,
    description="Coordinates loan underwriting flow end-to-end.",
    instruction=(
        "You are the orchestrator for loan applications. For each application:\n"
        "1) Use data_ingestion_agent to fetch external data.\n"
        "2) Use risk_policy_agent to compute risk and policy_decision.\n"
        "3) Use report_agent to build a RiskReport.\n"
        "If the decision is approve or deny and rules clearly apply, mark as straight-through.\n"
        "Otherwise, mark as manual_review and include guidance for human underwriters."
    ),
    sub_agents=[diva_agent, risk_policy_agent, report_agent],
)
