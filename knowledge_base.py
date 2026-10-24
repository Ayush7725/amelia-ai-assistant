"""
Knowledge base for Amelia.

Everything here is demo content for a fictional "Demo Bank". Replace it with
your own bank's product information before using Amelia for real customers.
The chat backend injects KB_TEXT into the system prompt and checks answers
against it, so the assistant only talks about what is written here.
"""

DEMO_ACCOUNTS = """
## Accounts
- Savings Account: minimum monthly balance ₹10,000; interest 3.0% p.a., paid quarterly.
- Zero Balance Savings Account: no minimum balance; one free debit card; limited free branch transactions.
- Senior Citizen Savings Account: for customers aged 60+; minimum balance ₹5,000; doorstep banking on request.
- Current Account: for businesses; minimum monthly balance ₹25,000; unlimited online transfers.
- Opening an account: full Video KYC from the app or website with Aadhaar and PAN; takes about 10 minutes.
"""

DEMO_CARDS = """
## Cards
- Debit Card: issued free with every savings account; daily ATM limit ₹50,000; daily shopping limit ₹1,00,000.
- Credit Card: annual fee ₹500, waived on yearly spend above ₹1,00,000; 1 reward point for every ₹100 spent.
- Lost or stolen card: block it instantly in chat after logging in, or call customer care.
"""

DEMO_LOANS = """
## Loans
- Personal Loan: ₹50,000 to ₹25,00,000; tenure 12 to 60 months; interest from 10.5% p.a.; minimum salary ₹25,000 per month.
- Home Loan: up to 80% of the property value; tenure up to 30 years; interest from 8.5% p.a.
- Car Loan: up to 90% of the on-road price; tenure up to 7 years; interest from 9.0% p.a.
- Eligibility: age 21 to 60, a stable income, and a CIBIL score of 700 or above (recommended).
"""

DEMO_DEPOSITS = """
## Fixed Deposits
- Tenure 7 days to 10 years; minimum deposit ₹5,000.
- Interest from 3.5% (short tenures) to 7.0% p.a. (1 to 3 years); senior citizens get an extra 0.5%.
"""

DEMO_DIGITAL = """
## Digital Banking and Security
- Services in chat: balance check, last transactions, fund transfer, card blocking, cheque validation, Video KYC.
- UPI, NEFT, RTGS and IMPS transfers are available in the mobile app.
- Demo Bank will NEVER ask for your full card number, CVV, OTP or PIN in chat, by call or by email.
"""

DEMO_SUPPORT = """
## Customer Support
- Customer care (demo): 1800-000-0000, available 24x7.
- Email (demo): support@demobank.example
- You can ask to talk to a human agent at any time during the chat.
"""

KB_TEXT = f"""
# DEMO BANK - KNOWLEDGE BASE (sample content)

{DEMO_ACCOUNTS}

{DEMO_CARDS}

{DEMO_LOANS}

{DEMO_DEPOSITS}

{DEMO_DIGITAL}

{DEMO_SUPPORT}
"""
