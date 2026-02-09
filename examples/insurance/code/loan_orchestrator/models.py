from pydantic import BaseModel, Field
from typing import List, Optional, Literal, Dict

class LoanApplication(BaseModel):
    application_id: str
    customer_id: str
    name: str
    ssn_last4: str
    income: float
    employer: str
    loan_amount: float
    purpose: str
    channel: Literal["web", "mobile"]
    submitted_at: str  # ISO timestamp

class CreditReport(BaseModel):
    bureau: Literal["Equifax", "TransUnion", "Experian"]
    score: int
    delinquencies: int
    utilization: float
    inquiries_12m: int
    trade_lines: int

class EmploymentVerification(BaseModel):
    employer: str
    status: Literal["verified", "unverified", "mismatch"]
    salary_range: str
    tenure_months: int

class FraudSignals(BaseModel):
    fraud_score: float
    flags: List[str] = []

class BankAnalysis(BaseModel):
    avg_monthly_income: float
    avg_monthly_expenses: float
    nsf_events_12m: int

class RiskScore(BaseModel):
    score: float = Field(ge=0, le=100)
    model_version: str

class Decision(BaseModel):
    decision: Literal["approve", "deny", "manual_review"]
    reason_summary: str
    policy_version: str
    is_straight_through: bool

class ReasoningStep(BaseModel):
    step: str
    details: str
    evidence: Dict[str, str]

class RiskReport(BaseModel):
    application: LoanApplication
    credit: Optional[CreditReport]
    employment: Optional[EmploymentVerification]
    fraud: Optional[FraudSignals]
    bank: Optional[BankAnalysis]
    risk_score: Optional[RiskScore]
    decision: Optional[Decision]
    reasoning_chain: List[ReasoningStep]
    artifacts: Dict[str, str] = {}
