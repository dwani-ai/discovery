Loan Underwriter

python3.10 -m venv venv
source venv/bin/activate

pip install google-adk pydantic fastapi uvicorn requests python-dotenv


cat > .env << 'EOF'
CREDIT_BUREAU_URL=http://localhost:8000/mock/credit
EMPLOYMENT_API_URL=http://localhost:8000/mock/employment
FRAUD_API_URL=http://localhost:8000/mock/fraud
BANK_API_URL=http://localhost:8000/mock/bank
POLICY_SERVICE_URL=http://localhost:8000/mock/policy
RISK_MODEL_URL=http://localhost:8000/mock/risk
EOF
