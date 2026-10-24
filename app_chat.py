import os
import hashlib
import json
import logging
import asyncio
import httpx
import uuid
import cv2
import re
import shutil
from PIL import Image
import pytesseract
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from dotenv import load_dotenv
import base64
import numpy as np
from datetime import datetime, timedelta
# Optional heavy AI deps - loaded lazily so the server can start without them
try:
    import easyocr
except Exception:
    easyocr = None
try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

# Configure Tesseract - adjust path if needed
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Knowledge Base Import (ensure knowledge_base.KB_TEXT exists)
try:
    from knowledge_base import KB_TEXT
except Exception:
    KB_TEXT = ""

load_dotenv()

# --- CONFIGURATION ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")  # optional: Groq / Gemini / OpenRouter / Ollama
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# Core-banking API (auth, balance, transfers, cards, cheque debit). See README for the contract.
BANK_API_BASE_URL = os.getenv("BANK_API_BASE_URL", "http://localhost:9000/api").rstrip("/")
# Cheque validation: bank name printed on the cheque and accepted MICR city-code prefixes
CHEQUE_BANK_NAME = os.getenv("CHEQUE_BANK_NAME", "DEMO BANK")
MICR_PREFIXES = tuple(x.strip() for x in os.getenv("MICR_PREFIXES", "632,633").split(",") if x.strip())
if not OPENAI_API_KEY:
    print("Error: OPENAI_API_KEY is missing in .env file.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Cheque Validation Configuration ---
CHEQUE_DIR = os.path.join(BASE_DIR, "uploads", "cheques")
os.makedirs(CHEQUE_DIR, exist_ok=True)
print(f"Cheque storage directory: {CHEQUE_DIR}")
print(f"Absolute path: {os.path.abspath(CHEQUE_DIR)}")

# ---Add By Dinesh --- store for documents ---
KYC_DOCUMENT_DIR = os.path.join(BASE_DIR, "uploads", "kyc_documents")
os.makedirs(KYC_DOCUMENT_DIR, exist_ok=True)

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | CHAT | %(message)s")
log = logging.getLogger("CHAT")

# OpenAI async client
client = (
    AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    if OPENAI_API_KEY
    else None
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Video KYC Configuration ---
KYC_VIDEO_DIR = os.path.join(BASE_DIR, "uploads", "kyc_videos")
os.makedirs(KYC_VIDEO_DIR, exist_ok=True)
print(f"KYC Video storage directory: {KYC_VIDEO_DIR}")

# KYC validation settings
KYC_DURATION_SECONDS = 10  # 10-second video
KYC_VALIDATION_RULES = {
    "min_face_size": 2500,  # Minimum face pixel area
    "max_face_size": 250000,  # Maximum face pixel area
    "min_brightness": 50,  # Minimum brightness level
    # Laplacian variance. Normal laptop webcams measure ~30-80, so the old limit of 100 called
    # every real webcam video "blurry"; below 20 a frame is genuinely smeared.
    "min_sharpness": 20,
    "stability_threshold": 0.8,  # Face stability score
}

active_customers = {}
active_agents = {}
client_lead_map = {}
client_to_lead = {}
account_intake_by_client = {}  # {client_id: {"first_name":..., "last_name":..., "address":...}}ayushh

# ========== NEW: Authentication State Management ==========
authenticated_clients = {}  # {client_id: {"authenticated": bool, "last4": str, "phone": str, "dob": str, "timestamp": datetime}}
pending_protected_operations = {}  # {client_id: {"operation": str, "original_request": str}}

def client_in_human_mode(cid: str) -> bool:
    try:
        return cid in client_to_lead and client_to_lead[cid] is not None
    except Exception:
        return False

# --- SYSTEM INSTRUCTIONS ---
# MODIFIED: Added strict rules against empty/dummy tables
SYSTEM_INSTRUCTIONS = f"""
You are Demo Bank AI.
Your goal is to assist users using the Knowledge Base below.

!!! CRITICAL: NUMERIC DATA FORMATTING !!!
1. ALWAYS display currency balances in numeric format with symbols (e.g., "$10,231.39").
2. NEVER write out numbers or amounts in words (e.g., NEVER say "ten thousand dollars").
3. When asking for digits (like the last 4 digits of a card), ALWAYS use digits (e.g., "last 4 digits").

!!! CRITICAL: NO DUMMY TABLES OR PLACEHOLDERS !!!
1. YOU ARE STRICTLY FORBIDDEN from generating tables with placeholder text like "Aspect", "Description", "Info", or "Detail".
2. NEVER generate a table that has empty rows or generic labels.
3. If the information is not found in the Knowledge Base, DO NOT create a table at all. Say: "I'm sorry, I don't have that information in my records."
4. Every single cell in a table MUST contain specific, factual data from the Knowledge Base.

!!! CRITICAL: SPEECH SPEED & PACING !!!
1. Speak at a steady, calm pace. Imagine pausing slightly between thoughts.
2. Insert natural breaks with phrases like 'Let me just check that for you.'
3. **STRICT RULE**: DO NOT use hyphens (-), dashes, or extra spaces between letters to simulate pacing (e.g., NO "H-e-l-l-o"). Use standard punctuation.
4. When reading numbers, keep them as digits (1, 2, 3) so the system can process them correctly.

!!! CRITICAL: CONTEXT ISOLATION & MEMORY RESET !!!
1. If the user asks for a "Balance," you MUST NOT repeat, summarize, or include any previous "Transaction" data in your response. 
2. If the user asks for "Transactions," you MUST NOT repeat the "Balance" data from a previous turn unless specifically asked.
3. Treat every new tool call as a "Memory Reset." Use ONLY the data from the most recent tool output to formulate your answer.
4. DO NOT provide a "summary" of the conversation history. Respond only to the current user request.

==================== SERVICES AVAILABLE ====================
1. Balance Enquiry (requires security verification)
2. Cheque Validation (upload cheque image)
3. Transaction Processing
4. Transaction History
5. General Banking Queries
==================== KNOWLEDGE BASE START ====================
{KB_TEXT}
==================== KNOWLEDGE BASE END ======================

CRITICAL PROTOCOLS (FOLLOW STRICTLY):

1. **STRICT TOOL PRIORITY (INTENT MAPPING):**
   - If the user mentions "Balance", "Transactions", "Statement", or "Block Card", you MUST assume they want to perform an action on *their* account.
   - **DO NOT** ask clarifying questions like "Are you asking about general transaction limits or your specific account?".
   - **ALWAYS** assume they want to use a TOOL (check_balance, get_last_transactions, etc.).

2. **CRITICAL: NO DUMMY DATA / NO HALLUCINATIONS:**
   - You are **STRICTLY FORBIDDEN** from generating fake, dummy, or example data.
   - **NEVER** show a table with "Grocery Store", "Online Shopping", "Uber", or "Netflix" unless the API specifically returned those strings.
   - **API IS THE ONLY TRUTH:** If the `get_last_transactions` tool returns an empty list or fails, say "I cannot access your transaction history at this time." DO NOT make up a list.
   - Your response must come **100% from the tool output**, not your own imagination.

3. LANGUAGE POLICY:

Default Language: English

Rules:
1. Always respond in the same language as the users most recent message.
2. Hindi input ? Hindi response.
3. English input ? English response.
4. Mixed Hindi-English input ? Mirror the mixture.
5. Unknown or unsupported language ? Respond in English.
6. Do not introduce additional languages unless explicitly requested.

4. **SECURITY PROTOCOL   CENTRAL AUTHENTICATION:**
   For the following operations, authentication is REQUIRED and MUST be performed FIRST:
   - Balance Enquiry (check_balance)
   - Card Blocking (block_card)
   - View Last Transactions (get_last_transactions)
   - Create Transaction (create_transaction)

   **CRITICAL AUTHENTICATION FLOW:**
   - **Step 1:** If user asks for any of the above, check if they are authenticated.
   - **Step 2:** If NOT authenticated, **STOP**. Do NOT ask for the "Account Number" yet.
   - **Step 3:** IMMEDIATELY ask for the 3 Mandatory Verification Fields:
     1. Registered Mobile Number
     2. Last 4 digits of account number
     3. Date of Birth (DD-MM-YYYY)
   - **Step 4:** Call `authenticate_user`.
   - **Step 5:** Only AFTER authentication is successful, proceed to ask for operation-specific details (like the *Full* Account Number for transactions).

5. **SECURITY PROTOCOL - TRANSACTION HISTORY:**
   - **Auth:** Authenticate FIRST.
   - **Input:** Ask for **FULL** customer account number ONLY AFTER auth.
   - **Verification**: You are FORBIDDEN from showing transactions for an account number that does not end in the digits the user provided during authentication.

6. **SAVINGS ACCOUNT HANDLING (CRITICAL CLARIFICATION):**
   - If the user asks about "Savings Account" (e.g., "tell me about savings account", "savings account details"):
     - **STOP.** Do NOT read from the Knowledge Base immediately.
     - **STOP.** Do NOT ask for details to open an account immediately.
     - **YOU MUST ASK:** "Are you looking for information on our savings account features, or would you like to open a new savings account?"
   - **PATH A (Features):** If they say "features" or "info", use the Knowledge Base.
   - **PATH B (Open Account):** If they say "open account", ONLY THEN ask for: First Name, Last Name, Mobile, Email, and Address.

7. **CONTEXT ISOLATION (NO BLEED-OVER):**
   - If the user changes the topic (e.g., from "My Transactions" to "What is Demo Bank?"), **FORGET** the previous tool output.
   - Do NOT repeat the transaction list.
   - Treat the new question as a fresh query using the Knowledge Base.

8. **SECURITY PROTOCOL   CARD BLOCKING:**
   - Authenticate FIRST.
   - Ask ONLY for:  last 4-digit card number and card type.
   - Call `block_card`.

9. **VIDEO KYC:**
    - If user asks for KYC, send the "kyc_trigger" JSON response immediately.

10. **STRICT TOOL DEPENDENCY:** - If the user asks a question that requires a tool, you are FORBIDDEN from generating a text response on your own.
    - You MUST call the specific API tool.

11. SECURITY PROTOCOL  TRANSACTION:
If the user requests a transaction (e.g., "perform a transaction", "send money", "transfer funds"):

- You MUST NOT refuse the request.
- You MUST NOT redirect the user to customer support or a branch.
- Treat this as a privileged action handled by you.

If any of the following are missing, you MUST ask for them one by one:
- sender_account_number
- bank_name
- customer_account_number
- customer_name
- amount

Do NOT guess or assume values.
If the user provides multiple details at once, store them and ask ONLY for missing ones.
Once all four details are collected, you MUST immediately call the create_transaction tool.

12. **ABSOLUTE TOOL FAILURE RULE (HIGHEST PRIORITY OVERRIDE):**
   - If any operation requires a tool and:
       1) the tool returns empty
       2) the tool returns null
       3) the tool fails
       4) the tool is unavailable
       5) or tool output is missing fields
   - You MUST NOT generate any example, placeholder, assumed, or formatted data.
   - You MUST NOT create tables, lists, or summaries from your own knowledge.
   - You MUST ONLY respond with:

     "I cannot access the requested information at this time. Please try again later."

   - This rule OVERRIDES all other instructions, including helpfulness.
   - Under NO circumstances should you fabricate banking data.

12 . You must sound like a trained, premium banking relationship manager , calm, intelligent, empathetic, and confident.
   
   ABSOLUTE FALLBACK FAILURE RULE (NON-NEGOTIABLE):
   
   If the requested information is NOT found in:
   - API tool output
   - Knowledge Base text
   
   Then you MUST:
   - NOT infer
   - NOT assume
   - NOT reconstruct
   - NOT summarize into lists or tables
   - NOT format into rows, columns, bullets, or headings
   - NOT use prior knowledge
   - NOT use general banking knowledge
   - NOT attempt to be helpful by guessing
   
   Your ONLY allowed response is a plain sentence such as:
   
   "I'm sorry, I don't have information about that at the moment."
   or
   "I don't currently have the details for this request."
   
   You MUST stop after this response.   

"""

# --- TOOL DEFINITION ---
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "authenticate_user",
            "description": "Authenticates the user. REQUIRED before performing any other action. Inputs: mobile_number, last4, dob, mpin.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mobile_number": {"type": "string", "description": "Registered Mobile Number"},
                    "last4": {"type": "string", "description": "Last 4 digits of account number"},
                    "dob": {"type": "string", "description": "Date of Birth (DD-MM-YYYY)"},
                    "mpin": {"type": "string", "description": "4-digit MPIN"}
                },
                "required": ["mobile_number", "last4", "dob", "mpin"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_balance",
            "description": "Checks the account balance. REQUIRES AUTHENTICATION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "last4": {"type": "string", "description": "Last 4 digits of account number"},
                    "mobile_number": {"type": "string", "description": "Mobile number"},
                    "dob": {"type": "string", "description": "Date of Birth (DD-MM-YYYY)"}
                },
                "required": ["last4", "mobile_number", "dob"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_transaction",
            "description": "Create a transaction record via the core-banking API. REQUIRES AUTHENTICATION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "bank_name": {"type": "string", "description": "Bank name"},
                    "customer_name": {"type": "string", "description": "Customer name"},
                    "sender_account_number": {"type": "string", "description": "Sender account number"},
                    "customer_account_number": {"type": "string", "description": "Customer account number"},
                    "amount": {"type": "number", "description": "Amount to debit/credit"}
                },
                "required": ["bank_name", "customer_name", "customer_account_number", "amount"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_last_transactions",
            "description": "Fetch last N transactions. REQUIRES AUTHENTICATION and FULL ACCOUNT NUMBER.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_account_number": {"type": "string", "description": "Customer FULL account number"},
                    "limit": {"type": "integer", "description": "Number of transactions (default 5)"}
                },
                "required": ["customer_account_number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "block_card",
            "description": "Block/deactivate a customer's card. REQUIRES AUTHENTICATION.",
            "parameters": {
                "type": "object",
                "properties": {
                    "card_number": {"type": "string", "description": "last 4-digit card number (mandatory)"},
                    "card_type": {"type": "string", "description": "Card type: debit/credit/mastercard"},
                    "confirm": {"type": "boolean", "description": "Must be true to proceed"}
                },
                "required": ["card_number", "card_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "transfer_call",
            "description": "Transfer to human agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "phone_number": {
                        "type": "string",
                        "description": "Customer phone number"
                    }
                },
                "required": ["phone_number"]
            }
        }
    }
]

# VAD SETTINGS
VAD_PARAMS = {
    "type": "server_vad",
    "threshold": 0.6,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500
}


# ========== NEW: Central Authentication Function ==========
async def authenticate_user_central(last4: str, phone: str, dob: str, mpin: str) -> tuple[bool, str]:
    """
    Central authentication against the core-banking auth endpoint
    Returns: (success: bool, message: str)
    """
    url = f"{BANK_API_BASE_URL}/auth"
    payload = {
        "last4": last4,
        "mobile": phone,
        "dob": dob,
        "mpin": mpin
    }
    log.info(f"Central Authentication Request: last4={last4}, mobile=***{phone[-2:] if phone else ''}")
    
    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.post(url, json=payload, timeout=8.0)
            try:
                data = resp.json()
            except Exception:
                log.warning(f"Non-JSON response from auth API: {resp.text}")
                return False, "Authentication service error. Please try again."
            
            log.info(f"Authentication API Response: {data}")
            
            # Check if authentication was successful
            if isinstance(data, dict):
                # NEW LOGIC: Check the explicit 'success' key from the JSON response first
                api_success = data.get("success")
                speak_msg = data.get("speak", "")
                
                # Check if API explicitly says success is True
                if api_success is True:
                    return True, " Authentication successful! You now have access to all banking services."
                
                # If success is False or missing, handle it as a failure
                else:
                    # Fallback to checking speak_msg if api_success isn't clear, 
                    # but prioritize the API's own success flag
                    error_msg = speak_msg or "Authentication failed. Please check your details."
                    return False, error_msg
            
            return False, "Authentication failed. Please try again."
            
        except Exception as e:
            log.error(f"Central Authentication API Error: {e}")
            return False, "System error during authentication. Please try again."

def is_client_authenticated(client_id: str) -> bool:
    """Check if a client is authenticated"""
    return client_id in authenticated_clients and authenticated_clients[client_id].get("authenticated", False)


def set_client_authenticated(client_id: str, last4: str, phone: str, dob: str, mpin: str = None):
    """Mark a client as authenticated"""
    authenticated_clients[client_id] = {
        "authenticated": True,
        "last4": last4,
        "phone": phone,
        "dob": dob,
        "mpin": mpin,
        "timestamp": datetime.now()
    }
    log.info(f"Client {client_id} authenticated successfully")


def get_client_auth_details(client_id: str) -> dict:
    """Get stored authentication details for a client"""
    if client_id in authenticated_clients:
        return authenticated_clients[client_id]
    return {}


# --- Cheque Validation Functions ---

def is_valid_layout(image_path: str) -> bool:
    """Layer 1: Checks whether image resembles a cheque layout."""
    try:
        img = Image.open(image_path)
        width, height = img.size
        aspect_ratio = width / height
        log.info(f"Testing mode: Image {width}x{height}, Aspect {aspect_ratio:.2f}")
        return width > 100 and height > 100
    except Exception as e:
        log.error(f"Layout validation error: {e}")
        return False


def extract_text_from_cheque(image_path: str) -> str:
    """Layer 2: Extracts text from cheque using OCR."""
    try:
        image = cv2.imread(image_path)
        if image is None:
            log.error(f"Cannot read image: {image_path}")
            return ""

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        config = "--oem 3 --psm 6"
        text = pytesseract.image_to_string(gray, config=config)
        log.info(f"OCR Raw Output for {os.path.basename(image_path)}:\n{text}")
        return text.upper()
    except Exception as e:
        log.error(f"OCR extraction error: {e}")
        return ""


def has_bank_name(text: str) -> bool:
    # Letters may be split or misread by OCR, so allow spaces between them
    pattern = r"\s*".join(re.escape(ch) for ch in CHEQUE_BANK_NAME.replace(" ", ""))
    return bool(re.search(pattern, text, re.IGNORECASE))


def extract_valid_micr(text: str) -> str | None:
    log.info(f"Looking for MICR in text: {text[:100]}...")
    digits_only = re.sub(r"\D", "", text)
    log.info(f"Digits only: {digits_only}")
    for i in range(len(digits_only) - 8):
        micr = digits_only[i:i+9]
        if micr.startswith(MICR_PREFIXES):
            log.info(f"Found MICR: {micr}")
            return micr
    log.info("No valid MICR found")
    return None

def extract_account_from_cheque_text(text: str) -> str | None:
    if not text:
        return None

    t = text.upper()

    # STRICT match for A/C NO pattern first
    m = re.search(r"A/C\s*NO\.?\s*[:\-]?\s*(\d{6,20})", t)
    if m:
        return m.group(1)

    # fallback only if strict fails
    digit_runs = re.findall(r"\d{8,20}", t)  # increase min length to 8
    for run in digit_runs:
        if run.startswith(MICR_PREFIXES):
            continue
        return run

    return None

# Cheque debit endpoint (override with CHEQUE_DEBIT_API_URL)
DEBIT_API_URL = os.getenv("CHEQUE_DEBIT_API_URL", f"{BANK_API_BASE_URL}/cheque-debit")

async def debit_account_via_api(account_number: str, amount: float, trace_id: str | None = None) -> dict:
    """
    Call the external PHP API to debit the given account number by amount.
    Returns API JSON as dict. Timeout + error handling.
    """
    payload = {"account_number": str(account_number), "amount": float(amount)}
    if trace_id:
        payload["trace_id"] = trace_id
    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.post(DEBIT_API_URL, json=payload, timeout=10.0)
            log.info(f"DEBIT PAYLOAD: {payload}")

            try:
                data = resp.json()
            except Exception:
                return {"success": False, "message": "Invalid JSON from debit API", "raw": resp.text}
            if isinstance(data, dict):
                return data
            return {"success": False, "message": "Unexpected response from debit API"}
        except Exception as e:
            log.error(f"debit_account_via_api error: {e}")
            return {"success": False, "message": f"Exception calling debit API: {e}"}
        
def extract_amount(text: str) -> float | None:
    text = text.replace(",", "")
    patterns = [
        r"\$\s*(\d+(\.\d{1,2})?)",
        r"\u20B9\s*(\d+(?:\.\d{1,2})?)",
        r"RUPEES\s*(\d+(?:\.\d{1,2})?)",
        r"RS\.?\s*(\d+(?:\.\d{1,2})?)",
        r"AMOUNT[:\s]*(\d+(?:\.\d{1,2})?)",
        r"=\s*(\d+(?:\.\d{1,2})?)"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            log.info(f"Found amount with pattern '{pattern}': {match.group(1)}")
            try:
                return float(match.group(1))
            except Exception:
                return None
    log.info("No amount found in text")
    return None

def add_empathy_prefix(user_text: str, response_text: str) -> str:
    """
    Adds empathy only when needed.
    For normal greetings / general info queries, it returns response_text unchanged.
    """
    try:
        text = (user_text or "").lower().strip()
        resp = (response_text or "").strip()

        if not resp:
            return response_text

        # 1) Greetings / small talk -> NO empathy prefix
        if text in {"hi", "hello", "hey", "hii", "hlo", "good morning", "good evening", "good afternoon"}:
            return response_text

        # 2) General info queries (like "tell me about demo bank") -> NO empathy prefix
        # Keep this simple: if it's clearly an info query and not money/error/kyc/card
        info_triggers = ["tell me about", "about", "what is", "who is", "information", "overview", "services", "features"]
        if any(t in text for t in info_triggers) and not any(k in text for k in [
            "balance", "transaction", "money", "transfer", "payment",
            "error", "failed", "problem", "issue", "not working", "wrong",
            "kyc", "verify", "verification", "document", "upload", "identity",
            "block", "lost", "stolen", "card"
        ]):
            return response_text

        # 3) Money related -> empathy
        if any(k in text for k in ["balance", "transaction", "money", "transfer", "payment"]):
            empathy = (
                "I completely understand — anything related to your money is important, "
                "and I want to make sure this is handled correctly for you."
            )
            return empathy + "\n\n" + response_text

        # 4) Errors -> empathy
        if any(k in text for k in ["error", "failed", "problem", "issue", "not working", "wrong"]):
            empathy = (
                "I'm really sorry you're experiencing this. I know it can be frustrating, "
                "but I'll do my best to get this sorted out for you."
            )
            return empathy + "\n\n" + response_text

        # 5) KYC -> empathy
        if any(k in text for k in ["kyc", "verify", "verification", "document", "upload", "identity"]):
            empathy = (
                "I understand this process can feel a bit sensitive, and I appreciate your patience. "
                "I'll guide you through it step by step."
            )
            return empathy + "\n\n" + response_text

        # 6) Card -> empathy
        if any(k in text for k in ["block", "lost", "stolen", "card"]):
            empathy = (
                "I'm glad you reached out right away — securing your account is very important, "
                "and I'll help you take care of this safely."
            )
            return empathy + "\n\n" + response_text
        
        # ✅ DEFAULT: generic empathy for unrelated queries
        empathy = "I understand what you're looking for, and I'm here to assist you in the best way possible."
        return empathy + "\n\n" + response_text

    except Exception:
        return response_text

def validate_cheque_rules(extracted_text: str) -> dict:
    results = {
        "bank_name_found": False,
        "valid_micr": None,
        "amount_found": None,
        "all_valid": False
    }
    results["bank_name_found"] = has_bank_name(extracted_text)
    results["valid_micr"] = extract_valid_micr(extracted_text)
    results["amount_found"] = extract_amount(extracted_text)
    results["all_valid"] = (
        results["bank_name_found"] and
        results["valid_micr"] is not None and
        results["amount_found"] is not None and
        results["amount_found"] > 0
    )
    return results


def is_valid_cheque(image_path: str) -> tuple[bool, object]:
    log.info(f"Validating cheque: {image_path}")
    if not is_valid_layout(image_path):
        log.info("Layout validation failed")
        return False, "Layout validation failed: Image does not resemble a cheque"
    extracted_text = extract_text_from_cheque(image_path)
    log.info(f"Extracted OCR Text:\n{extracted_text}")
    if not extracted_text.strip():
        log.info("OCR extraction failed")
        return False, "OCR extraction failed: No text detected"
    validation_results = validate_cheque_rules(extracted_text)
    if not validation_results["all_valid"]:
        log.info("Cheque rule validation failed")
        return False, "Cheque rule validation failed"
    log.info("Demo Bank cheque validated successfully")
    log.info(f"   MICR: {validation_results['valid_micr']}")
    log.info(f"   Amount: {validation_results['amount_found']}")
    return True, validation_results


async def validate_cheque_image(file: UploadFile):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filename = os.path.basename(file.filename).replace(" ", "_")
    file_path = os.path.join(CHEQUE_DIR, f"{timestamp}_{safe_filename}")

    try:
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        log.info(f"Cheque saved to: {os.path.abspath(file_path)}")

        # Run cheque validation
        is_valid, result = is_valid_cheque(file_path)

        if not is_valid:
            return {
                "success": False,
                "message": f"{result}",
                "details": None,
                "file_path": file_path,
                "timestamp": timestamp
            }

        # Base response (temporary message, will update after debit)
        response = {
            "success": True,
            "message": "Demo Bank cheque validated successfully!",
            "details": result,
            "file_path": file_path,
            "timestamp": timestamp
        }

        # --- Extract account number and attempt debit ---
        try:
            raw_text = extract_text_from_cheque(file_path)
            acct = extract_account_from_cheque_text(raw_text)

            if not acct:
                response["debit_attempt"] = {
                    "success": False,
                    "message": "Could not detect account number from cheque OCR."
                }
                return response

            amount = result.get("amount_found")

            if not amount or amount <= 0:
                response["debit_attempt"] = {
                    "success": False,
                    "message": "Amount not found or zero; skipping debit."
                }
                return response

            trace_id = str(uuid.uuid4())

            debit_res = await debit_account_via_api(
                account_number=acct,
                amount=amount,
                trace_id=trace_id
            )

            # Normalize response
            if not isinstance(debit_res, dict):
                debit_res = {"success": False, "message": str(debit_res)}

            response["debit_attempt"] = {
                "account_number": acct,
                "amount": amount,
                "trace_id": trace_id,
                "api_result": debit_res
            }

            # --- If debit successful, update final message ---
            if debit_res.get("success"):

                previous_balance = float(debit_res.get("previous_balance") or 0)
                new_balance = float(debit_res.get("new_balance") or 0)

                response["message"] = (
                    f"Demo Bank cheque validated successfully!\n"
                    f"Account Number: {acct}\n"
                    f"Amount Credited: ${amount:,.2f}\n"
                    f"Previous Balance: ${previous_balance:,.2f}\n"
                    f"Current Balance: ${new_balance:,.2f}"
                )

            else:
                response["message"] += (
                    f"\nCredit Failed: {debit_res.get('message', 'Unknown error')}"
                )

        except Exception as e:
            log.exception("Error while attempting to debit account from cheque")
            response["debit_attempt"] = {
                "success": False,
                "message": f"Exception: {str(e)}"
            }

        return response

    except Exception as e:
        log.error(f"Cheque validation error: {e}")
        return {
            "success": False,
            "message": f"Error processing cheque: {str(e)}",
            "details": None,
            "file_path": None,
            "timestamp": timestamp
        }


# --- Video KYC Functions ---

def get_clear_image(image_path: str) -> str | None:
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(gray)
    final_img = cv2.fastNlMeansDenoising(enhanced_img, h=5)
    proc_path = image_path.replace('.jpg', '_proc.jpg').replace('.png', '_proc.png')
    cv2.imwrite(proc_path, final_img)
    return proc_path


# --- Lightweight face engine (OpenCV DNN: YuNet detector + SFace recognizer) ---
# No torch / TensorFlow, so it is safe to run on a normal laptop.
YUNET_MODEL = os.path.join(BASE_DIR, "models", "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(BASE_DIR, "models", "face_recognition_sface_2021dec.onnx")
SFACE_COSINE_THRESHOLD = 0.363  # OpenCV SFace same-identity threshold

_face_detector = None
_face_recognizer = None


def get_face_detector():
    global _face_detector
    if _face_detector is None:
        try:
            _face_detector = cv2.FaceDetectorYN.create(YUNET_MODEL, "", (320, 320), score_threshold=0.6)
            log.info("YuNet face detector loaded")
        except Exception as e:
            log.error(f"Face detector load failed: {e}")
            _face_detector = False
    return _face_detector or None


def get_face_recognizer():
    global _face_recognizer
    if _face_recognizer is None:
        try:
            _face_recognizer = cv2.FaceRecognizerSF.create(SFACE_MODEL, "")
            log.info("SFace recognizer loaded")
        except Exception as e:
            log.error(f"Face recognizer load failed: {e}")
            _face_recognizer = False
    return _face_recognizer or None


def detect_faces(img_bgr):
    det = get_face_detector()
    if det is None or img_bgr is None:
        return []
    h, w = img_bgr.shape[:2]
    if h < 10 or w < 10:
        return []
    try:
        det.setInputSize((w, h))
        _, faces = det.detect(img_bgr)
    except Exception as e:
        log.error(f"Face detection error: {e}")
        return []
    return list(faces) if faces is not None else []


def _best_face(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: float(f[2]) * float(f[3]))


def ocr_image_text(image_path: str) -> str:
    """OCR a document with Tesseract (light, no heavy models)."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return ""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        return pytesseract.image_to_string(gray) or ""
    except Exception as e:
        log.error(f"OCR error: {e}")
        return ""


def _face_match_score(img_a_path, img_b_path):
    """Compare the largest face in each image with SFace.
    Returns (cosine_score, matched) or (None, None) if a face is missing."""
    a = cv2.imread(img_a_path)
    b = cv2.imread(img_b_path)
    if a is None or b is None:
        return None, None
    fa = _best_face(detect_faces(a))
    fb = _best_face(detect_faces(b))
    if fa is None or fb is None:
        return None, None
    rec = get_face_recognizer()
    if rec is None:
        return None, None
    try:
        align_a = rec.alignCrop(a, fa)
        align_b = rec.alignCrop(b, fb)
        score = float(rec.match(rec.feature(align_a), rec.feature(align_b), cv2.FaceRecognizerSF_FR_COSINE))
        return score, score >= SFACE_COSINE_THRESHOLD
    except Exception as e:
        log.error(f"Face match error: {e}")
        return None, None


# --- Indian documents: Aadhaar (12 digits + Verhoeff checksum), PAN, passport ---
_VERHOEFF_D = [[0,1,2,3,4,5,6,7,8,9],[1,2,3,4,0,6,7,8,9,5],[2,3,4,0,1,7,8,9,5,6],[3,4,0,1,2,8,9,5,6,7],
               [4,0,1,2,3,9,5,6,7,8],[5,9,8,7,6,0,4,3,2,1],[6,5,9,8,7,1,0,4,3,2],[7,6,5,9,8,2,1,0,4,3],
               [8,7,6,5,9,3,2,1,0,4],[9,8,7,6,5,4,3,2,1,0]]
_VERHOEFF_P = [[0,1,2,3,4,5,6,7,8,9],[1,5,7,6,2,8,3,0,9,4],[5,8,0,3,7,9,6,1,4,2],[8,9,1,6,0,4,3,5,2,7],
               [9,4,5,3,1,2,6,8,7,0],[4,2,8,6,5,7,3,9,0,1],[2,7,9,3,8,0,6,4,1,5],[7,0,4,6,9,1,3,2,5,8]]


def valid_aadhaar(number: str) -> bool:
    if not re.fullmatch(r"[2-9]\d{11}", number or ""):
        return False
    c = 0
    for i, ch in enumerate(reversed(number)):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][int(ch)]]
    return c == 0


def find_indian_id(text: str):
    """(type, number shown to people) for an Aadhaar or PAN in OCR text, else (None, None).
    Aadhaar is masked to the last 4 digits; it is never returned or logged in full."""
    for m in re.finditer(r"(?<!\d)(\d{4})[ \-]?(\d{4})[ \-]?(\d{4})(?!\d)", text or ""):
        number = "".join(m.groups())
        if valid_aadhaar(number):
            return "Aadhaar", "XXXX XXXX " + number[-4:]
    m = re.search(r"\b([A-Z]{5}\d{4}[A-Z])\b", (text or "").upper())
    if m:
        return "PAN", m.group(1)
    return None, None


_ADDRESS_STOP = {"the", "and", "for", "road", "street", "st", "ave", "avenue", "near", "opp", "house", "flat",
                 "floor", "no", "number", "india", "usa", "state", "city", "district", "dist", "post", "po"}


def _words(text):
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(w) >= 3 and w not in _ADDRESS_STOP}


def check_address_proof(address_proof_path, document_name, entered):
    """Read the proof of address: it must show the customer's name and the address they typed."""
    text = ocr_image_text(address_proof_path) if address_proof_path else ""
    if len(_words(text)) < 3:
        return {"status": "Failed", "reason": "the proof of address could not be read"}
    names = [n for n in (document_name, f"{entered.get('first_name', '')} {entered.get('last_name', '')}") if n]
    first_names = {n.split()[0].lower() for n in names if n.split() and n != "Not Detected"}
    name_found = any(fn in text.lower() for fn in first_names)
    shared = _words(entered.get("address", "")) & _words(text)
    address_found = len(shared) >= 2
    status = "Verified" if name_found and address_found else "Failed"
    reason = None
    if not name_found:
        reason = "the proof of address does not show your name"
    elif not address_found:
        reason = "the address on the proof does not match the address you entered"
    return {"status": status, "name_found": name_found, "address_match": address_found,
            "matched_words": sorted(shared), "reason": reason}


def names_match(entered_name: str, document_name: str) -> bool:
    """Every word of the name the customer typed must appear in the name on the document."""
    entered_words = re.findall(r"[a-z]+", (entered_name or "").lower())
    document_words = set(re.findall(r"[a-z]+", (document_name or "").lower()))
    return bool(entered_words) and document_name != "Not Detected" and all(w in document_words for w in entered_words)


# ayushh 13-02-2026: Added more robust document verification with OCR and face matching
def verify_documents(national_id_path, passport_path, address_proof_path, video_frame_path):
    results = {
        "id_data": {"number": "Not Detected", "type": "USA ID/SSN", "quality": "OK"},
        "passport_data": {"number": "Not Detected", "quality": "OK"},
        "person_data": {"name": "Not Detected", "dob": "Not Detected"},
        "address_data": {"status": "Not Processed"},
        "face_match": False, "face_confidence": 0, "status": "Failed", "message": ""
    }
    try:
        if not video_frame_path or not (national_id_path or passport_path):
            raise Exception("Missing required paths.")
        results["_ocr"] = {}
        if national_id_path:
            id_text = ocr_image_text(national_id_path)
            results["_ocr"]["id"] = id_text
            ocr_lines = [l.strip() for l in id_text.splitlines() if l.strip()]
            blob = re.sub(r"\s+", " ", id_text).strip()

            # DOB: 12/02/2001 or 12-02-2001 or 2001-02-12
            dob = None
            m = re.search(r"\b(\d{2}[/-]\d{2}[/-]\d{4})\b", blob)
            if not m:
                m = re.search(r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b", blob)
            if m:
                dob = m.group(1)

            # Name: try line containing "Name"
            name = None
            for ln in ocr_lines:
                if re.search(r"\bname\b", ln, re.IGNORECASE):
                    cleaned = re.sub(r"(?i)\bname\b", "", ln).replace(":", " ").strip()
                    if len(cleaned) >= 3:
                        name = cleaned
                        break

            # fallback: choose name-like line (2+ words, mostly letters), skipping card headers
            # such as "GOVERNMENT OF INDIA" / "INCOME TAX DEPARTMENT"
            header_words = {"government", "india", "income", "tax", "department", "permanent", "account",
                            "number", "republic", "identification", "card", "unique", "authority", "address",
                            "dob", "birth", "male", "female", "passport", "election", "commission", "state"}
            if not name:
                candidates = []
                for ln in ocr_lines:
                    x = re.sub(r"[^A-Za-z\s]", " ", ln)
                    x = re.sub(r"\s+", " ", x).strip()
                    if header_words & set(x.lower().split()):
                        continue
                    if len(x.split()) >= 2 and len(x) >= 6:
                        candidates.append(x)
                if candidates:
                    name = max(candidates, key=len)

            results["person_data"]["name"] = name or "Not Detected"
            results["person_data"]["dob"]  = dob  or "Not Detected"






            text_blob = blob.upper()
            clean_blob = text_blob.replace(" ", "").replace("-", "").replace(".", "")
            indian_type, indian_number = find_indian_id(blob)
            wi_match = None if indian_type else re.search(r'[A-Z]\d{13}', clean_blob)
            dl_match = re.search(r'[A-Z]\d{7,9}', clean_blob)
            ssn_match = re.search(r'\b\d{9}\b', clean_blob)
            if indian_type:
                results["id_data"]["number"] = indian_number
                results["id_data"]["type"] = indian_type
            elif wi_match:
                results["id_data"]["number"] = wi_match.group()
                results["id_data"]["type"] = "Wisconsin State ID"
            elif dl_match:
                results["id_data"]["number"] = dl_match.group()
                results["id_data"]["type"] = "Driver's License"
            elif ssn_match:
                results["id_data"]["number"] = ssn_match.group()
                results["id_data"]["type"] = "SSN"
        if passport_path:
            pass_text = ocr_image_text(passport_path)
            results["_ocr"]["passport"] = pass_text
            pass_blob = pass_text.upper()
            second_type, second_number = find_indian_id(pass_blob)
            pass_match = None if second_type else re.search(r'\b[A-Z]\d{7}\b|\bC\d{8}\b|\b\d{9}\b|\b[A-Z0-9]{9}\b', pass_blob)
            if second_type:
                results["passport_data"]["number"] = second_number
                results["passport_data"]["type"] = second_type
            if pass_match:
                results["passport_data"]["number"] = pass_match.group()
                log.info(f"Passport Detected: {pass_match.group()}")
            elif not second_type:
                strict_blob = pass_blob.replace(" ", "")
                fallback_match = re.search(r'C\d{8}|\d{9}', strict_blob)
                if fallback_match:
                    results["passport_data"]["number"] = fallback_match.group()
        match_target = national_id_path if national_id_path else passport_path
        if match_target and video_frame_path:
            score, matched = _face_match_score(video_frame_path, match_target)
            if score is not None:
                results["face_match"] = bool(matched)
                results["face_confidence"] = round(score * 100, 2)
            else:
                results["message"] = "Could not detect a face in the video frame or the document."
        has_id = results["id_data"]["number"] != "Not Detected" or results["passport_data"]["number"] != "Not Detected"
        if results["face_match"] and has_id:
            results["status"] = "Verified"
            results["message"] = "Identity verified successfully."
        else:
            results["status"] = "Failed"
            results["message"] = results["message"] or "Face mismatch or ID number not recognized."
    except Exception as e:
        results["status"] = "Error"
        results["message"] = str(e)
    return results


# Utility helpers


def save_base64_image(base64_data, output_path):
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image_data = decode_frame_any_format(base64_data)
        if not image_data:
            return False
        with open(output_path, "wb") as f:
            f.write(image_data)
        return True
    except Exception as e:
        print(f"Image save failed: {e}")
        return False


def handle_kyc_documents(session_id, aadhaar_snapshot, pan_snapshot):
    result = {
        "documents_ok": False,
        "aadhaar_saved": False,
        "pan_saved": False,
        "paths": {}
    }
    if not aadhaar_snapshot or not pan_snapshot:
        return result
    session_dir = os.path.join(KYC_DOCUMENT_DIR, session_id)
    os.makedirs(session_dir, exist_ok=True)
    aadhaar_path = os.path.join(session_dir, "aadhaar.jpg")
    pan_path = os.path.join(session_dir, "pan.jpg")
    aadhaar_saved = save_base64_image(aadhaar_snapshot, aadhaar_path)
    pan_saved = save_base64_image(pan_snapshot, pan_path)
    if aadhaar_saved and pan_saved:
        result["documents_ok"] = True
        result["aadhaar_saved"] = True
        result["pan_saved"] = True
        result["paths"] = {"aadhaar": aadhaar_path, "pan": pan_path}
    return result


def decode_frame_any_format(frame: str) -> bytes | None:
    if not frame or not isinstance(frame, str):
        return None
    try:
        frame = re.sub(r"\s+", "", frame)
        if "," in frame:
            frame = frame.split(",", 1)[1]
        missing = len(frame) % 4
        if missing:
            frame += "=" * (4 - missing)
        return base64.b64decode(frame, validate=False)
    except Exception:
        return None


def _strip_markup_inline(s: str) -> str:
    """Remove markdown characters and special formatting symbols from inline text"""
    if not s:
        return s
    # Remove simple markdown characters that clutter UI
    s = re.sub(r"[_`]+", "", s)
    s = re.sub(r"\*+", "", s)  # Remove all asterisks
    s = re.sub(r"#+\s*", "", s)  # Remove hash/pound symbols with optional space
    # Clean up for any remaining special characters
    s = re.sub(r"[*#]+", "", s)
    return s


def _extract_json_object(text: str):
    """Parse a JSON object from model text, tolerating ```json fences and prose."""
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except Exception:
        pass
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end + 1])
        except Exception:
            return None
    return None


def _normalize_kb_text(s: str) -> str:
    """Normalize KB text by removing markdown and formatting characters"""
    if not s:
        return s
    # Remove code/markdown markers and convert headers/bullets to line breaks
    s = s.replace('\r', '\n')
    s = re.sub(r"[_`]+", "", s)
    s = re.sub(r"\*+", "", s)  # Remove all asterisks
    # Convert header markers to newline separators
    s = re.sub(r"#{1,6}\s*", "\n", s)  # Remove hash symbols
    # Convert ' - ' sequences to new lines for better row splitting
    s = re.sub(r"\s+-\s+", "\n- ", s)
    # Collapse multiple blank lines
    s = re.sub(r"\n{2,}", "\n", s)
    # Normalize spaces
    s = re.sub(r"[ \t]{2,}", " ", s)
    # Final pass to remove any remaining * or #
    s = re.sub(r"[*#]+", "", s)
    return s.strip()


def _is_from_kb(response: str, threshold: float = 0.06, min_matches: int = 5) -> bool:
    """Rudimentary check whether the response likely came from KB_TEXT by token overlap.
    Returns True if enough tokens from KB_TEXT appear in response.
    """
    try:
        if not response or not KB_TEXT:
            return False
        # tokenize words (alpha-only), lowercased
        def toks(s):
            return set(w.lower() for w in re.findall(r"[A-Za-z]{3,}", s))
        resp_tokens = toks(response)
        kb_tokens = toks(KB_TEXT)
        if not resp_tokens or not kb_tokens:
            return False
        inter = resp_tokens.intersection(kb_tokens)
        if len(inter) >= min_matches:
            # ratio relative to response tokens
            ratio = len(inter) / max(1, len(resp_tokens))
            return ratio >= threshold or len(inter) >= (min_matches * 1)
        return False
    except Exception:
        return False


def _user_requested_detailed(messages_list) -> bool:
    try:
        last_user = ''
        for m in reversed(messages_list):
            if m.get('role') == 'user':
                last_user = (m.get('content') or '')
                break
        if not last_user:
            return False
        return bool(re.search(r"\b(detail|detailed|in detail|explain more|full|comprehensive|more details)\b", last_user.lower()))
    except Exception:
        return False


def _parse_kb_blocks_to_table(text: str, detailed: bool = False):
    """Parse KB-style blocks into a table. Returns (columns, rows).
    Attempts to extract product/section headings and key:value pairs.
    """
    try:
        txt = _normalize_kb_text(text)
        # split on double newlines to get blocks
        blocks = [b.strip() for b in re.split(r"\n{2,}", txt) if b.strip()]
        rows = []
        col_keys = set()
        for blk in blocks:
            lines = [l.strip() for l in blk.split('\n') if l.strip()]
            if not lines:
                continue
            # heading detection (numbered like '1. Personal Finance')
            heading = ''
            if re.match(r"^\d+\.\s*", lines[0]):
                heading = re.sub(r"^\d+\.\s*", "", lines[0])
                rest = lines[1:]
            else:
                # if first line contains ':' it's likely a key:value, treat whole block as details
                if ':' in lines[0] and len(lines) == 1:
                    heading = ''
                    rest = lines
                else:
                    heading = lines[0]
                    rest = lines[1:]

            kv = {}
            if heading:
                kv['Product'] = heading
            for ln in rest:
                m = re.match(r"([^:]+):\s*(.+)", ln)
                if m:
                    k = m.group(1).strip()
                    v = m.group(2).strip()
                    kv[k] = v
                    col_keys.add(k)
                else:
                    # sentence-style content -> append to Notes
                    kv.setdefault('Notes', '')
                    if kv['Notes']:
                        kv['Notes'] += ' ' + ln
                    else:
                        kv['Notes'] = ln
            rows.append(kv)

        if not rows:
            return None, None

        # Build columns: Product first, then sorted kv keys, ensure Notes at end
        cols = []
        cols.append('Product')
        sorted_keys = sorted(k for k in col_keys if k.lower() != 'notes')
        cols.extend(sorted_keys)
        if 'Notes' in col_keys or any('Notes' in r for r in rows):
            cols.append('Notes')

        # If no kv keys found, fallback to single Info column
        if len(cols) == 1:
            cols = ['Product', 'Info']

        out_rows = []
        for r in rows:
            row = []
            for c in cols:
                val = r.get(c, '-')
                # Clean markdown and special characters from cell values
                val = re.sub(r"[_`*#]+", "", val)
                val = re.sub(r"\s{2,}", " ", val).strip()
                row.append(val or '-')
            out_rows.append(row)

        # limit rows unless detailed requested
        if not detailed and len(out_rows) > 12:
            out_rows = out_rows[:12]

        return cols, out_rows
    except Exception:
        return None, None


def analyze_video_frame(frame_base64: str) -> dict:
    try:
        frame_data = decode_frame_any_format(frame_base64)
        if not frame_data:
            return {"valid": False, "face_detected": False}
        nparr = np.frombuffer(frame_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {"valid": False}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(frame)
        results = {
            "valid": False,
            "face_detected": len(faces) > 0,
            "face_count": len(faces),
            "brightness": float(np.mean(gray)),
            "is_blurry": False
        }
        if len(faces) == 1:
            w, h = float(faces[0][2]), float(faces[0][3])
            face_area = int(w * h)
            results["face_size"] = face_area
            if (KYC_VALIDATION_RULES["min_face_size"] <= face_area <= KYC_VALIDATION_RULES["max_face_size"]):
                if results["brightness"] >= KYC_VALIDATION_RULES["min_brightness"]:
                    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                    results["is_blurry"] = laplacian_var < KYC_VALIDATION_RULES["min_sharpness"]
                    results["valid"] = not results["is_blurry"]
        return results
    except Exception as e:
        log.error(f"Frame analysis error: {e}")
        return {"valid": False}


# Liveness: the customer is asked to look straight, then turn the head left and right. A photo
# held to the camera can be moved, but its nose never moves relative to its eyes, so the head
# turn (yaw) is what proves a live person. Yaw = nose offset from the eye midpoint, in eye widths.
LIVENESS_YAW_DELTA = 0.15  # needed on BOTH sides of the starting pose
LIVENESS_MIN_FACE_FRAMES = 0.6  # share of frames that must show exactly one face


def frame_yaw(frame_bgr):
    faces = detect_faces(frame_bgr)
    if len(faces) != 1:
        return None, len(faces)
    f = faces[0]
    rex, rey, lex, ley, nx = (float(v) for v in (f[4], f[5], f[6], f[7], f[8]))
    eye_dist = max(1e-6, ((lex - rex) ** 2 + (ley - rey) ** 2) ** 0.5)
    return (nx - (rex + lex) / 2) / eye_dist, 1


def validate_kyc_video(frames_data: list) -> dict:
    results = {"total": len(frames_data), "valid": 0, "faces": 0, "is_valid": False}
    yaws, extra_faces = [], 0
    for f in frames_data:
        res = analyze_video_frame(f)
        if res.get("valid"):
            results["valid"] += 1
        if res.get("face_detected"):
            results["faces"] += 1
        frame = cv2.imdecode(np.frombuffer(decode_frame_any_format(f) or b"", np.uint8), cv2.IMREAD_COLOR)
        yaw, count = frame_yaw(frame) if frame is not None else (None, 0)
        if count > 1:
            extra_faces += 1
        if yaw is not None:
            yaws.append(yaw)
    n = max(1, len(frames_data))
    baseline = sorted(yaws[:3])[len(yaws[:3]) // 2] if yaws else 0.0  # "look straight" at the start
    turned_one_way = bool(yaws) and max(yaws) - baseline >= LIVENESS_YAW_DELTA
    turned_other_way = bool(yaws) and baseline - min(yaws) >= LIVENESS_YAW_DELTA
    results["head_turn"] = {
        "both_sides": turned_one_way and turned_other_way,
        "yaw_range": round(max(yaws) - min(yaws), 3) if yaws else 0.0,
    }
    problems = []
    if len(yaws) < n * LIVENESS_MIN_FACE_FRAMES:
        problems.append("your face was not clearly visible in most of the video")
    if extra_faces:
        problems.append("more than one face was seen")
    if results["valid"] <= n * 0.3:
        problems.append("the video was too dark, blurry or the face too small")
    if not results["head_turn"]["both_sides"]:
        problems.append("we could not see you turn your head left and right")
    results["problems"] = problems
    results["is_valid"] = not problems
    return results


def process_video_liveness(video_path: str):
    log.info(f"Processing Video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    extracted_frames = []
    if not cap.isOpened():
        return [], {"is_valid": False, "error": "Video open failed"}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        # Browser webm often has no frame count: read it all, then sample evenly so the head
        # turns later in the video are seen (it used to keep only the first frames).
        log.warning("Using manual frame extraction fallback...")
        all_frames = []
        while len(all_frames) < 1800:
            ret, frame = cap.read()
            if not ret:
                break
            all_frames.append(frame)
        step = max(1, len(all_frames) // 15)
        for frame in all_frames[::step]:
            _, buffer = cv2.imencode('.jpg', frame)
            extracted_frames.append(base64.b64encode(buffer).decode('utf-8'))
    else:
        step = max(1, total_frames // 15)
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                _, buffer = cv2.imencode('.jpg', frame)
                extracted_frames.append(base64.b64encode(buffer).decode('utf-8'))
    cap.release()
    validation_results = validate_kyc_video(extracted_frames)
    return extracted_frames, validation_results


async def save_kyc_video(video_path, user_id, session_id, doc_paths):
    try:
        frames_list, validation = process_video_liveness(video_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata = {
            "session_id": session_id,
            "user_id": user_id,
            "is_valid_kyc": validation.get("is_valid", False),
            "files": {"video": video_path, "docs": doc_paths},
            "metrics": validation,
            "timestamp": timestamp
        }
        meta_path = os.path.join(KYC_VIDEO_DIR, f"meta_{user_id}_{timestamp}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        return metadata, frames_list
    except Exception as e:
        log.error(f"Save KYC Video Error: {e}")
        return {"error": str(e)}, []


# --- HELPER FUNCTION (API HIT) ---
async def fetch_balance_from_api(last4, mobile, dob, mpin=None):
    url = f"{BANK_API_BASE_URL}/auth"
    payload = {"last4": last4, "mobile": mobile, "dob": dob, "mpin": mpin}
    log.info(f"Balance API Request: last4={last4}")
    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.post(url, json=payload, timeout=8.0)
            try:
                data = resp.json()
            except Exception:
                log.warning(f"Non-JSON response from balance API: {resp.text}")
                data = {}
            log.info(f"API Response: {data}")
            if isinstance(data, dict):
                return data.get("speak", "I could not verify your details. Please try again.")
            else:
                return "I could not verify your details. Please try again."
        except Exception as e:
            log.error(f"API Error: {e}")
            return "System error while checking balance."


async def create_transaction_in_api(bank_name, customer_name, sender_account_number, customer_account_number, amount, trace_id=None):
    url = f"{BANK_API_BASE_URL}/transfer"
    payload = {"bank_name": bank_name, "customer_name": customer_name, "sender_account_number": sender_account_number, "customer_account_number": customer_account_number, "amount": amount}
    if trace_id:
        payload["trace_id"] = trace_id
    log.info(f"Transaction API Request{f' ({trace_id})' if trace_id else ''}: {payload}")
    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.post(url, json=payload, timeout=10.0)
            log.info(f"HTTP {resp.status_code} FROM transaction API: {resp.text}")
            try:
                data = resp.json()
            except Exception:
                data = {}
            log.info(f"Transaction API Response: {data}")
            if isinstance(data, dict) and data.get("success"):
                new_balance = data.get("new_balance")
                prev_balance = data.get("previous_balance")
                speak_text = (f"Transaction successful. Your previous balance was {prev_balance}$. Your current balance is {new_balance}$." )
                return {"success": True, "speak": speak_text, "balance": new_balance, "previous_balance": prev_balance, "transaction_id": data.get("transaction_id"), "data": data}
            return {"success": False, "speak": data.get("message") or data.get("error") or "Transaction failed due to server error.", "data": data}
        except Exception as e:
            log.error(f"Transaction API Error: {e}")
            return {"success": False, "speak": "System error while creating transaction.", "data": {}}


async def fetch_last_transactions_from_api(customer_account_number, limit=5, trace_id=None):
    url = f"{BANK_API_BASE_URL}/transactions"
    payload = {"customer_account_number": customer_account_number, "limit": limit}
    if trace_id:
        payload["trace_id"] = trace_id
    log.info(f"GetLastTx API Request{f' ({trace_id})' if trace_id else ''}: {payload}")
    async with httpx.AsyncClient() as http_client:
        try:
            resp = await http_client.post(url, json=payload, timeout=8.0)
            log.info(f"HTTP {resp.status_code} FROM GetLastTx API: {resp.text}")
            try:
                data = resp.json()
            except Exception:
                data = {}
            log.info(f"GetLastTx API Response: {data}")
            if isinstance(data, dict) and data.get("success") and isinstance(data.get("last_transactions"), list):
                return {"speak": "Fetched last transactions.", "transactions": data.get("last_transactions"), "data": data}
            else:
                return {"speak": data.get("message") or data.get("error") or "No transactions found."}
        except Exception as e:
            log.error(f"GetLastTx API Error: {e}")
            return {"speak": "System error while fetching transactions. Please try again."}


async def block_card_in_api(card_number, card_type, trace_id=None):
    log.critical("block_card_in_api() CALLED ")
    if not trace_id:
        trace_id = str(uuid.uuid4())
    card_clean = ''.join(filter(str.isdigit, str(card_number)))
    if len(card_clean) != 4:
        return {"speak": "Card number must be exactly 4 digits.", "success": False, "trace_id": trace_id}
    url = f"{BANK_API_BASE_URL}/block-card"
    payload = {"card_id": card_clean, "card_type": card_type, "trace_id": trace_id, "action": "BLOCKED"}
    log.info(f"BlockCard API Request ({trace_id}): {payload}")
    async with httpx.AsyncClient() as client_http:
        try:
            resp = await client_http.post(url, json=payload, timeout=10.0)
            log.info(f"HTTP {resp.status_code} FROM BlockCard API: {resp.text}")
            try:
                data = resp.json()
            except Exception as e:
                log.error(f"JSON parse error from BlockCard API: {e} - body: {resp.text}")
                data = {"success": False, "message": "Invalid JSON from backend", "raw": resp.text}
            log.info(f"BlockCard API Response ({trace_id}): {data}")
            if data.get("success"):
                return {"speak": f"Your {card_type} has been blocked successfully.", "success": True, "data": data, "trace_id": trace_id}
            else:
                return {"speak": data.get("message") or data.get("error") or "Failed to block the card.", "success": False, "data": data, "trace_id": trace_id}
        except Exception as e:
            log.error(f"BlockCard API Error ({trace_id}): {e}")
            return {"speak": "System error while blocking the card.", "success": False, "data": {}, "trace_id": trace_id}


# ==========================================
# GLOBAL STATE
# ==========================================
pending_transfers = []
client_histories = {}


def _msg_field(msg, name, default=None):
    if isinstance(msg, dict):
        return msg.get(name, default)
    return getattr(msg, name, default)


def _merge_system_messages(msgs):
    """Fold every system message into the first one.
    Gemini's OpenAI-compatible endpoint keeps only the LAST system message,
    which would silently drop SYSTEM_INSTRUCTIONS (incl. the Knowledge Base)."""
    system_parts = [m["content"] for m in msgs if isinstance(m, dict) and m.get("role") == "system" and m.get("content")]
    others = [m for m in msgs if not (isinstance(m, dict) and m.get("role") == "system")]
    if not system_parts:
        return others
    return [{"role": "system", "content": "\n\n".join(system_parts)}] + others


def _fmt_money_usd(v):
    try:
        num = float(v)
        return f"${num:,.2f}"
    except Exception:
        return str(v) if v is not None else ""


def build_transactions_table_payload(transactions, title="Last 5 transactions"):
    columns = ["Date", "Txn ID", "Type", "Amount"]
    rows = []
    for tx in (transactions or [])[:12]:
        date = tx.get("date") or tx.get("created_at") or tx.get("transaction_date") or tx.get("timestamp") or ""
        txn_id = tx.get("transaction_id") or tx.get("id") or tx.get("txn_id") or ""
        ttype = tx.get("type") or tx.get("transaction_type") or tx.get("dr_cr") or ""
        amt = _fmt_money_usd(tx.get("amount") or tx.get("amt") or tx.get("value") or "")
        rows.append([str(date), str(txn_id), str(ttype), str(amt)])
    return {"type": "table", "title": title, "columns": columns, "rows": rows}


def build_transactions_text(transactions):
    lines = ["Here are your last transactions:"]
    for i, tx in enumerate((transactions or [])[:5], start=1):
        date = tx.get("date") or tx.get("created_at") or tx.get("transaction_date") or tx.get("timestamp") or ""
        amt = _fmt_money_usd(tx.get("amount") or tx.get("amt") or tx.get("value") or "")
        txn_id = tx.get("transaction_id") or tx.get("id") or tx.get("txn_id") or ""
        lines.append(f"{i}) {date} | {amt}  | ID: {txn_id}")
    return "\n".join(lines)


def _extract_account_number(text: str) -> str | None:
    if not text:
        return None
    digits = re.sub(r"\D", "", text)
    if 10 <= len(digits) <= 20:
        return digits
    return None


def _extract_limit(text: str, default: int = 5) -> int:
    if not text:
        return default
    m = re.search(r"\b(last|recent)\s+(\d{1,2})\b", text.lower())
    if m:
        try:
            n = int(m.group(2))
            return max(1, min(n, 20))
        except Exception:
            return default
    m2 = re.search(r"\b(\d{1,2})\s*(transactions|txns|txn|tx)\b", text.lower())
    if m2:
        try:
            n = int(m2.group(1))
            return max(1, min(n, 20))
        except Exception:
            return default
    return default


def _is_last_transactions_intent(text: str) -> bool:
    t = (text or "").lower()
    triggers = [
        "last transaction", "last transactions",
        "last 5 transaction", "last 5 transactions", "last five transaction", "last five transactions",
        "recent transaction", "recent transactions",
        "last 5 txn", "last five txn", "last 5 txns", "last five txns",
        "my transactions", "show transactions", "show my transactions","spending",
        "show my last spendings", "what did i pay yesterday?", "show recent payments",
        "any money came to my account?", "show last 5 entries", "mini receipt",
        "transaction history", "mini statement"
    ]
    return any(k in t for k in triggers)


# ========== NEW: Helper to parse auth details from user message ==========
def parse_auth_details(text: str) -> dict:
    """
    Parse authentication details from user text.
    Returns dict with last4, phone, and dob if all found.   
    """
    result = {"last4": None, "phone": None, "dob": None}
    
    # Extract last 4 digits (look for 4 consecutive digits)
    last4_match = re.search(r'\b(\d{4})\b', text)
    if last4_match:
        result["last4"] = last4_match.group(1)
    
    # Extract phone number (10+ digits)
    phone_match = re.search(r'\b(\d{10,})\b', text)
    if phone_match:
        result["phone"] = phone_match.group(1)
    
    # Extract DOB (DD-MM-YYYY or DD/MM/YYYY format)
    dob_match = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{2,4})', text)
    if dob_match:
        result["dob"] = dob_match.group(1).replace("/", "-")
    
    return result


def _detect_account_types_query(text: str) -> bool:
    """Detect if user is asking about account types or Demo Bank info"""
    text_lower = text.lower()
    account_triggers = [
        "account type", "types of account", "what accounts",
        "account details", "tell me about accounts",
        "savings account", "current account", "account options",
        "different accounts", "account information",
        "demo bank", "about demo bank", "tell me about demo bank"
    ]
    return any(trigger in text_lower for trigger in account_triggers)


# --- WEBSOCKET ENDPOINT ---
@app.websocket("/ws/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    client_id = str(uuid.uuid4())
    active_customers[client_id] = websocket
     # Store this client_id in session or send to frontend
    await websocket.send_json({
        "type": "client_id",
        "client_id": client_id
    })

    log.info(f"Chat Client Connected: {client_id}")
    function_call_map = {}
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTIONS}]

    # Local state flags
    pending_transfer_phone = False
    greeting_completed = False
    awaiting_account_details = False
    pending_tx_table_choice = False
    pending_tx_transactions = None
    pending_tx_account = None
    pending_tx_limit = 5
    pending_tx_account_prompt = False
    pending_tx_requested_limit = 5
    pending_tx_prefers_table = False

    # ========== FIXED: Clear transaction cache properly ==========
    last_tx_cache_account = None
    last_tx_cache_transactions = None
    last_tx_cache_limit = 5
    last_tx_cache_time = None

    # ========== NEW: Authentication state flags ==========
    pending_authentication = False
    awaiting_mpin = False
    pending_auth_last4 = None
    pending_auth_phone = None
    pending_auth_dob = None

    try:
        while True:
            # Initialize per-turn flags to avoid UnboundLocalError
            skip_post_tool_llm = False

            try:
                raw = await websocket.receive_text()
                log.info(f"RAW_IN (truncated 300): {raw[:300]}")
                payload = {}
                try:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict):
                        payload = parsed
                except Exception:
                    payload = {}
                user_text = ""
                if isinstance(payload.get("text"), str) and payload.get("text").strip():
                    user_text = payload.get("text").strip()
                else:
                    user_text = (raw or "").strip()
                if not user_text:
                    continue
                log.info(f"User: {user_text[:300]}")
                
                user_text_lower = user_text.lower()
                
                # ========== FIXED: Handle authentication flow FIRST before anything else ==========
                if pending_authentication:
                    if awaiting_mpin:
                        # Stage 2: MPIN already requested - authenticate now
                        mpin_match = re.search(r'\b(\d{4})\b', user_text)
                        if not mpin_match:
                            await websocket.send_json({"type": "bot_chunk", "text": "Please provide your 4-digit MPIN to complete verification."})
                            await websocket.send_json({"type": "bot_done"})
                            continue

                        auth_success, auth_message = await authenticate_user_central(
                            pending_auth_last4,
                            pending_auth_phone,
                            pending_auth_dob,
                            mpin_match.group(1)
                        )

                        if auth_success:
                            set_client_authenticated(
                                client_id,
                                pending_auth_last4,
                                pending_auth_phone,
                                pending_auth_dob,
                                mpin_match.group(1)
                            )
                            awaiting_mpin = False
                            pending_auth_last4 = pending_auth_phone = pending_auth_dob = None
                            await websocket.send_json({"type": "bot_chunk", "text": auth_message})
                            pending_authentication = False

                            # ========== CRITICAL FIX: Resume the protected operation immediately ==========
                            if client_id in pending_protected_operations:
                                pending_op = pending_protected_operations.pop(client_id)
                                operation_type = pending_op.get("operation")
                                original_request = pending_op.get("original_request")

                                log.info(f"Authentication successful - resuming {operation_type} request")

                                # Set user_text to the original request and fall through to normal processing
                                user_text = original_request
                                user_text_lower = user_text.lower()
                                # DO NOT call messages.append here - it will be called below
                            else:
                                await websocket.send_json({"type": "bot_done"})
                                continue
                        else:
                            await websocket.send_json({"type": "bot_chunk", "text": auth_message})
                            await websocket.send_json({"type": "bot_done"})
                            pending_authentication = False
                            awaiting_mpin = False
                            pending_auth_last4 = pending_auth_phone = pending_auth_dob = None
                            # Clear the pending operation on auth failure
                            if client_id in pending_protected_operations:
                                pending_protected_operations.pop(client_id)
                            continue
                    else:
                        # Stage 1: collect the 3 details, then ask for MPIN
                        auth_details = parse_auth_details(user_text)

                        if auth_details["last4"] and auth_details["phone"] and auth_details["dob"]:
                            pending_auth_last4 = auth_details["last4"]
                            pending_auth_phone = auth_details["phone"]
                            pending_auth_dob = auth_details["dob"]
                            awaiting_mpin = True
                            await websocket.send_json({"type": "bot_chunk", "text": "Thanks. Now please enter your 4-digit MPIN to complete verification."})
                            await websocket.send_json({"type": "bot_done"})
                            continue
                        else:
                            # Not all details provided
                            await websocket.send_json({"type": "bot_chunk", "text": "Please provide all three details: Last 4 digits of account, phone number, and date of birth (DD-MM-YYYY)."})
                            await websocket.send_json({"type": "bot_done"})
                            continue

                # ========== CRITICAL FIX: Only append to messages ONCE ==========
                messages.append({"role": "user", "content": user_text})
                
                # ========== FIXED: Check if user needs authentication for protected operations ==========
                protected_keywords = {
                    "balance": ["balance", "check balance", "account balance", "my balance", "current balance", "what is my balance"],
                    "block_card": ["block card", "block my card", "deactivate card", "disable card", "stop card"],
                    "transaction_history": ["last transaction", "transaction history", "recent transactions", "show transactions","last 5 transaction", "last 5 transactions", "last five transaction", "last five transactions", "recent transaction", "recent transactions", "last 5 txn", "last five txn", "last 5 txns", "last five txns", "my transactions", "show transactions", "show my transactions", "mini statement"],
                    "create_transaction": ["transfer", "send money", "create transaction", "make payment", "perform transaction", "do a transaction", "make a transfer", "i want to perform a transaction"]
                }
                
                needs_auth = False
                operation_type = None
                 # Detect human transfer request (NO AUTH REQUIRED)
                is_human_transfer = (
                    "transfer" in user_text_lower
                    and any(x in user_text_lower for x in ["agent", "human", "person", "representative"])
                )
                for op_type, keywords in protected_keywords.items():
                    if any(keyword in user_text_lower for keyword in keywords):
                        # Skip authentication if user wants human agent
                        if op_type == "create_transaction" and is_human_transfer:
                           continue
                        needs_auth = True
                        operation_type = op_type
                        break
                
                # ========== CRITICAL: Authenticate FIRST before asking operation details ==========
                if needs_auth and not is_client_authenticated(client_id) and not pending_authentication:
                    # Determine operation type for user message
                    operation_display = {
                        "balance": "balance check",
                        "block_card": "card blocking",
                        "transaction_history": "transaction history",
                        "create_transaction": "transaction"
                    }.get(operation_type, "this operation")
                    
                    # Store the operation and original request
                    pending_protected_operations[client_id] = {
                        "operation": operation_display,
                        "original_request": user_text
                    }
                    
                    # Start authentication flow - collect details, then MPIN separately
                    pending_authentication = True
                    awaiting_mpin = False
                    pending_auth_last4 = pending_auth_phone = pending_auth_dob = None
                    await websocket.send_json({"type": "bot_chunk", "text": f"To proceed with your {operation_display} request, I need to authenticate you first.\n\nPlease provide:\n1. Last 4 digits of your account number\n2. Your registered phone number\n3. Your date of birth (DD-MM-YYYY)\n\nYou can send all three in one message. I will then ask for your 4-digit MPIN."})
                    await websocket.send_json({"type": "bot_done"})
                    continue
                
                account_keywords = [
                    "i want to open a savings account",
                    "i want to open an savings account",
                    "i want to open a saving account",
                    "i want to open an saving account",
                    "open savings account",
                    "open savings account for me",
                    "open saving account",
                    "open saving account for me",
                    "open account",
                    "create account",
                    "new account"
                ]
                # KYC requests use the same flow: details first, then the KYC button (the model
                # used to answer with raw JSON {"kyctrigger": true}).
                kyc_keywords = ["kyc", "video kyc", "verify my identity", "identity verification"]
                wants_kyc = any(kw in user_text_lower for kw in kyc_keywords)
                if awaiting_account_details:
                    text = user_text.strip()

                    # FIXED: Better regex patterns that stop at the next field
                    def extract(pattern, text):
                        # Look for pattern and capture until next field or end of string
                        matches = re.search(pattern, text, re.IGNORECASE)
                        if matches:
                            return matches.group(1).strip()
                        return None

                    details = {
                        "first_name": extract(r"first\s*name\s*[:\-]\s*([^\n]+?)(?=\s+(?:last\s*name|mobile|email|address|$))", text),
                        "last_name": extract(r"last\s*name\s*[:\-]\s*([^\n]+?)(?=\s+(?:first\s*name|mobile|email|address|$))", text),
                        "mobile": extract(r"mobile\s*[:\-]\s*(\d{10})", text),
                        "email": extract(r"email\s*[:\-]\s*([^\s]+@[^\s]+)", text),
                        "address": extract(r"address\s*[:\-]\s*(.+)", text)  # Address can be the rest
                    }

                     # If the above fails, try simpler patterns
                    if not details["first_name"]:
                        # Try to capture just the name after "First Name:"
                        fn_match = re.search(r"first\s*name\s*[:\-]\s*([A-Za-z\s]+)", text, re.IGNORECASE)
                        if fn_match:
                            details["first_name"] = fn_match.group(1).strip().split()[0]  # Take first word only
                    
                    if not details["last_name"]:
                        ln_match = re.search(r"last\s*name\s*[:\-]\s*([A-Za-z\s]+)", text, re.IGNORECASE)
                        if ln_match:
                            details["last_name"] = ln_match.group(1).strip().split()[0]  # Take first word only

                    missing = [k for k, v in details.items() if not v]

                    if missing:
                        await websocket.send_json({
                            "type": "bot_chunk",
                            "text": "Missing Format: " + ", ".join(missing) + ". Please resend all details with proper format."
                        })

                        await websocket.send_json({"type": "bot_done"})
                        continue

                      # Clean the extracted values
                    details["first_name"] = details["first_name"].split()[0] if details["first_name"] else ""  # Take first word only
                    details["last_name"] = details["last_name"].split()[0] if details["last_name"] else ""    # Take first word only

                    account_intake_by_client[client_id] = {
                        "first_name": details["first_name"],
                        "last_name": details["last_name"],
                        "address": details["address"],
                        "mobile": details.get("mobile", ""),
                        "email": details.get("email", "")
                    }

                     # ALSO store with the SAME client_id as the key (this is what gets sent in KYC)
                    account_intake_by_client[client_id] = account_intake_by_client[client_id]

                     # ADD THIS DEBUG LINE TO VERIFY IT'S STORED
                    print(f"STORED account details for client_id: {client_id}")
                    print(f"Details: {account_intake_by_client[client_id]}")

                    awaiting_account_details = False

                    await websocket.send_json({
                        "type": "kyc_trigger",
                        "message": "Ok, please keep 3 documents ready: your ID card (Aadhaar / National ID), your PAN card or Passport, and a proof of address. Then you will record a short face video: look straight, then turn your head left and right.",
                        "show_kyc": True,
                        "duration": 10
                    })

                    await websocket.send_json({"type": "bot_done"})
                    continue


                # 2Start account opening flow (ask details)
                if (not awaiting_account_details) and (wants_kyc or any(kw in user_text_lower for kw in account_keywords)):
                    awaiting_account_details = True

                    await websocket.send_json({
                        "type": "bot_chunk",
                        "text": (
                            ("To start your Video KYC, please share the following details in one message:\n\n"
                             if wants_kyc else
                             "To open your savings account, please share the following details in one message:\n\n") +
                            "First Name:\n"
                            "Last Name:\n"
                            "Mobile:\n"
                            "Email:\n"
                            "Address:\n\n"
                            
                        )
                    })
                    await websocket.send_json({"type": "bot_done"})
                    continue

                # ========== ENHANCED: Detect queries that should force table format ==========
                TABLE_KEYWORDS = ["table format", "in table", "as a table", "tabular", "table form"]
                wants_table = any(k in user_text_lower for k in TABLE_KEYWORDS)
                
                # Force table for account types and Demo Bank queries
                if _detect_account_types_query(user_text):
                    wants_table = True
                    log.info("Account types/Demo Bank query detected - forcing table format")
                
                # Treat requests that ask for 'types', 'list', or 'which types' as table-intent
                if not wants_table:
                    if re.search(r"\b(type|types|list|which types|what types|show types)\b", user_text_lower):
                        wants_table = True
                # Also treat short 'tell me about <category>' or direct category queries as table-intent
                if not wants_table:
                    kb_topics = r"\b(cards|card|accounts|account|loans|loan|products|services|fees|benefits)\b"
                    # Match variations like 'tell me about X', 'tell me details about X', 'tell me detailed about X'
                    if re.search(r"tell me(?: .*?)?about\s+" + kb_topics, user_text_lower) or re.fullmatch(kb_topics, user_text_lower.strip()):
                        wants_table = True

                # ========== FIXED: Only use cached transactions if asking about TRANSACTIONS specifically ==========
                if wants_table and last_tx_cache_account is not None and last_tx_cache_transactions is not None:
                    # Only show cached transactions if user is asking about transactions
                    if _is_last_transactions_intent(user_text):
                        if len(last_tx_cache_transactions) == 0:
                            await websocket.send_json({"type": "bot_chunk", "text": f"There are no transactions available to show in a table for account {last_tx_cache_account}."})
                            await websocket.send_json({"type": "bot_done"})
                            continue
                        await websocket.send_json(build_transactions_table_payload(last_tx_cache_transactions, title=f"Last {last_tx_cache_limit} transactions"))
                        await websocket.send_json({"type": "bot_done"})
                        continue

                # Pending table confirmation
                if pending_tx_table_choice and pending_tx_transactions is not None:
                    yes_words = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "haan", "ha"}
                    no_words = {"no", "n", "nope", "nah", "na"}
                    u = user_text_lower.strip()
                    if u in yes_words or ("yes" in u) or ("table" in u):
                        await websocket.send_json(build_transactions_table_payload(pending_tx_transactions, title=f"Last {pending_tx_limit} transactions"))
                        await websocket.send_json({"type": "bot_done"})
                    elif u in no_words or ("no" in u):
                        text = build_transactions_text(pending_tx_transactions)
                        await websocket.send_json({"type": "bot_chunk", "text": text})
                        await websocket.send_json({"type": "bot_done"})
                    else:
                        await websocket.send_json({"type": "bot_chunk", "text": "Would you like this in table format(Yes/No)"})
                        await websocket.send_json({"type": "bot_done"})
                    pending_tx_table_choice = False
                    pending_tx_transactions = None
                    pending_tx_account = None
                    pending_tx_limit = 5
                    continue

                if last_tx_cache_transactions:
                    log.debug("Using cached transactions")

                # If waiting for account number
                if pending_tx_account_prompt:
                    acct = _extract_account_number(user_text)
                    if not acct:
                        await websocket.send_json({"type": "bot_chunk", "text": "Please provide your customer account number."})
                        await websocket.send_json({"type": "bot_done"})
                        continue
                    
                    # SECURITY CHECK: Ownership Verification
                    auth_details = get_client_auth_details(client_id)
                    stored_last4 = str(auth_details.get("last4", ""))
                    if not str(acct).endswith(stored_last4):
                        await websocket.send_json({"type": "bot_chunk", "text": "I cannot access the requested information. You can only view transactions for your own authenticated account."})
                        await websocket.send_json({"type": "bot_done"})
                        pending_tx_account_prompt = False
                        continue
                        
                    trace_id = str(uuid.uuid4())
                    limit = int(pending_tx_requested_limit or 5)
                    api_result = await fetch_last_transactions_from_api(customer_account_number=acct, limit=limit, trace_id=trace_id)
                    txs = api_result.get("transactions") or []
                    last_tx_cache_account = acct
                    last_tx_cache_transactions = txs
                    last_tx_cache_limit = limit
                    last_tx_cache_time = datetime.now()
                    if not txs:
                        await websocket.send_json({"type": "bot_chunk", "text": f"I couldn't find any transactions for account {acct}."})
                        await websocket.send_json({"type": "bot_done"})
                        pending_tx_account_prompt = False
                        continue
                    pending_tx_table_choice = True
                    pending_tx_transactions = txs
                    pending_tx_account = acct
                    pending_tx_limit = limit
                    pending_tx_account_prompt = False
                    await websocket.send_json({"type": "bot_chunk", "text": "Would you like this in table format(Yes/No)"})
                    await websocket.send_json({"type": "bot_done"})
                    continue

                # Detect last transactions intent before calling OpenAI
                if _is_last_transactions_intent(user_text):
                    limit = _extract_limit(user_text, default=5)
                    acct = _extract_account_number(user_text)
                    if not acct:
                        pending_tx_account_prompt = True
                        pending_tx_requested_limit = limit
                        pending_tx_prefers_table = wants_table
                        await websocket.send_json({"type": "bot_chunk", "text": "Please provide your customer account number."})
                        await websocket.send_json({"type": "bot_done"})
                        continue
                    
                    # SECURITY CHECK: Ownership Verification
                    auth_details = get_client_auth_details(client_id)
                    stored_last4 = str(auth_details.get("last4", ""))
                    if not str(acct).endswith(stored_last4):
                        await websocket.send_json({"type": "bot_chunk", "text": "I cannot access the requested information. You can only view transactions for your own authenticated account."})
                        await websocket.send_json({"type": "bot_done"})
                        continue

                    trace_id = str(uuid.uuid4())
                    api_result = await fetch_last_transactions_from_api(customer_account_number=acct, limit=limit, trace_id=trace_id)
                    txs = api_result.get("transactions") or []
                    last_tx_cache_account = acct
                    last_tx_cache_transactions = txs
                    last_tx_cache_limit = limit
                    last_tx_cache_time = datetime.now()
                    if not txs:
                        await websocket.send_json({"type": "bot_chunk", "text": f"I couldn't find any transactions for account {acct}."})
                        await websocket.send_json({"type": "bot_done"})
                        continue
                    pending_tx_table_choice = True
                    pending_tx_transactions = txs
                    pending_tx_account = acct
                    pending_tx_limit = limit
                    await websocket.send_json({"type": "bot_chunk", "text": "Would you like this in table format(Yes/No)"})
                    await websocket.send_json({"type": "bot_done"})
                    continue
                if client_in_human_mode(client_id):
                   connected_lead_id = None
                   for lead, mapped_client_id in client_lead_map.items():
                       if mapped_client_id == client_id:
                           connected_lead_id = lead
                           break
   
                   if connected_lead_id and connected_lead_id in active_agents:
                      try:
                          agent_ws = active_agents[connected_lead_id]
                          await agent_ws.send_json({"type": "customer_message", "text": user_text, "lead_id": connected_lead_id})
                          log.info(f"Relayed Customer text to Agent (Lead {connected_lead_id})")
                      except Exception as e:
                          log.error(f"Failed to relay to agent: {e}")
                      continue

                CHAT_TO_VOICE_KEYWORDS = ["switch to voice", "voice bot", "start voice", "call me", "voice mode", "switch to call"]
                if any(k in user_text_lower for k in CHAT_TO_VOICE_KEYWORDS):
                    log.info("Chat->Voice command detected, instructing frontend to switch to voice")
                    try:
                        await websocket.send_json({"type": "frontend_action", "action": "switch_to_voice", "reason": "user typed switch to voice", "action_id": str(uuid.uuid4()), "source": "chat"})
                        await websocket.send_json({"type": "bot_chunk", "text": "Switching to voice mode..."})
                        await websocket.send_json({"type": "bot_done"})
                    except Exception:
                        log.exception("Failed to send frontend_action switch_to_voice")
                        continue

                # Handle pending phone number collection for transfer
                if pending_transfer_phone:
                    phone_match = re.search(r'\b(\d{10,})\b', user_text)
                    if phone_match:
                        transfer_phone = phone_match.group(1)
                        pending_transfer_phone = False
                        
                        formatted_history = "--- PREVIOUS CHAT HISTORY ---\n"
                        for msg in messages:
                            role = _msg_field(msg, "role", "unknown")
                            content = _msg_field(msg, "content", "")
                            if role == "user":
                                formatted_history += f"User: {content}\n"
                            elif role == "assistant":
                                formatted_history += f"Bot: {content}\n"
                        formatted_history += "--------------------------------\n"
                        client_histories[client_id] = formatted_history
                        if client_id not in pending_transfers:
                            pending_transfers.append(client_id)
                        log.info(f"Client {client_id} added to pending_transfers queue with phone {transfer_phone}.")
                        await websocket.send_text(json.dumps({"type": "response.cancel"}))
                        await websocket.send_json({"type": "bot_chunk", "text": "Connecting you to a human agent... please stay on the page."})
                        await websocket.send_json({"type": "bot_done"})
                        await websocket.send_json({"type": "frontend_action", "action": "start_human_chat_call", "destination": "1001", "phone_number": transfer_phone})
                        continue
                    else:
                        await websocket.send_json({"type": "bot_chunk", "text": "Please provide a valid phone number (10 or more digits)."})
                        await websocket.send_json({"type": "bot_done"})
                        continue
                
                if "transfer" in user_text_lower and any(x in user_text_lower for x in ["agent", "human", "person"]):
                    log.info(f"TRANSFER REQUESTED by Client {client_id}")
                    pending_transfer_phone = True
                    await websocket.send_json({"type": "bot_chunk", "text": "Before I connect you to a human agent, please provide your phone number so the agent can reach you."})
                    await websocket.send_json({"type": "bot_done"})
                    continue
                upload_keywords = ["upload cheque", "upload a cheque", "want to upload cheque", "upload check", "validate cheque", "cheque", "scan cheque", "cheque upload", "check upload", "i want to upload"]
                if any(keyword in user_text_lower for keyword in upload_keywords):
                    upload_response = {"type": "upload_trigger", "message": "I can help you upload and validate a Demo Bank cheque! ", "show_upload": True}
                    await websocket.send_json(upload_response)
                    await websocket.send_json({"type": "bot_done"})
                    continue

                cheque_keywords = ["cheque", "validate", "image", "scan", "micr"]
                if any(keyword in user_text_lower for keyword in cheque_keywords):
                    cheque_info = (
                        "I can help you validate Demo Bank cheques! \n\n"
                        "You can upload a cheque image through the upload button () in the top-right corner.\n\n"
                        "I'll validate it using:\n"
                        "Layout analysis\n"
                        "Text extraction (OCR)\n"
                        "Demo Bank specific rules\n"
                        "MICR code verification (632/633 series)\n\n"
                        "Click the upload button to get started!"
                    )
                    await websocket.send_json({"type": "bot_chunk", "text": cheque_info})
                    await websocket.send_json({"type": "bot_done"})
                    continue

                if client_in_human_mode(client_id):
                    continue

                messages_for_turn = list(messages)
                
                # ========== CRITICAL FIX: Explicit Authentication Context ==========
                if is_client_authenticated(client_id):
                    auth_details = get_client_auth_details(client_id)
                    messages_for_turn.append({"role": "system", "content": f"""CRITICAL AUTHENTICATION STATUS:
User is FULLY AUTHENTICATED. Credentials: last4={auth_details.get('last4')}, phone={auth_details.get('phone')}, dob={auth_details.get('dob')}

FORCARD BLOCKING: Ask ONLY for the last 4-digit card number and type.
FOR BALANCES: Display numerically (e.g. $10,000.00).
DO NOT ask user to authenticate again."""})
                
                if wants_table:
                    messages_for_turn.append({"role": "system", "content": (
                        "User requested TABLE FORMAT.\n"
                        "Return ONLY valid JSON (no markdown, no extra text) in this schema:\n"
                        "{\"type\":\"table\",\"title\":\"string\",\"columns\":[],\"rows\":[],\"note\":\"string optional\"}\n"
                        "Rules:\n- rows <= 12\n- columns <= 6\n- every cell must be a string\n"
                        "- Remove ALL markdown characters (*, #, _, `) from all cell values\n"
                    )})

                try:
                    if wants_table:
                        # Do NOT expose tools here: the model otherwise emits a bogus "json" tool call
                        # which providers like Groq reject with 400 (tool_use_failed).
                        response = await client.chat.completions.create(model=OPENAI_MODEL, messages=_merge_system_messages(messages_for_turn), stream=False)
                    else:
                        response = await client.chat.completions.create(model=OPENAI_MODEL, messages=_merge_system_messages(messages_for_turn), tools=TOOLS, tool_choice="auto", stream=False)
                except Exception:
                    log.exception("OpenAI request failed")
                    await websocket.send_json({"type": "bot_chunk", "text": "System error: AI request failed."})
                    await websocket.send_json({"type": "bot_done"})
                    continue

                response_msg = None
                try:
                    if hasattr(response, "choices") and len(response.choices) > 0:
                        response_msg = response.choices[0].message
                    else:
                        log.warning("OpenAI response has no choices or unexpected shape.")
                except Exception:
                    log.exception("Error extracting response_msg from OpenAI response")

                final_text = (getattr(response_msg, "content", "") or "").strip()
                if wants_table and final_text:
                    try:
                        obj = _extract_json_object(final_text)
                        if isinstance(obj, dict) and obj.get("type") == "table":
                            
                            # ========== NEW GUARD: Detect Empty/Dummy Table ==========
                            # Check if the rows contain placeholder words like "Aspect" or "Description"
                            rows = obj.get("rows", [])
                            is_dummy = False
                            for row in rows:
                                if any(word in str(cell) for word in ["Aspect", "Description", "Info", "Detail"] for cell in row):
                                    is_dummy = True
                                    break
                            
                            if is_dummy or not rows:
                                log.warning("Detected dummy or empty table response. Skipping delivery.")
                                # We tell the LLM to try again in a system message (this loop continues)
                                messages.append({"role": "system", "content": "The table you generated contains dummy or placeholder text. Please provide factual information from the Knowledge Base or state if the data is unavailable."})
                                # To avoid infinite loops, we can allow one retry or just respond with text
                                bot_text = "I'm looking that up for you right now..."
                                await websocket.send_json({"type": "bot_chunk", "text": bot_text})
                                await websocket.send_json({"type": "bot_done"})
                                continue

                            # Clean all cell values of markdown
                            if "rows" in obj:
                                cleaned_rows = []
                                for row in obj["rows"]:
                                    cleaned_row = [_strip_markup_inline(str(cell)) for cell in row]
                                    cleaned_rows.append(cleaned_row)
                                obj["rows"] = cleaned_rows
                            if "columns" in obj:
                                obj["columns"] = [_strip_markup_inline(str(col)) for col in obj["columns"]]
                            if "title" in obj:
                                obj["title"] = _strip_markup_inline(str(obj["title"]))
                            # Record the reply so the next turn doesn't re-answer this question
                            messages.append({"role": "assistant", "content": json.dumps(obj)})
                            await websocket.send_json(obj)
                            await websocket.send_json({"type": "bot_done"})
                            continue
                    except Exception:
                        pass

                # =========================================================================
                # CASE A: Tool call from assistant
                # =========================================================================
                if response_msg and getattr(response_msg, "tool_calls", None):
                    log.info("AI Triggered Tool Call")
                    messages.append(response_msg)
                    
                    for tool_call in response_msg.tool_calls:
                        try:
                            tname = getattr(tool_call.function, "name", None)
                            args_raw = getattr(tool_call.function, "arguments", "{}")
                            try:
                                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                            except Exception:
                                args = {}

                            log.info(f"Executing Tool: {tname}")
                            output_content = "Error executing tool"
                            
                            protected_tools = ["check_balance", "create_transaction", "get_last_transactions", "block_card"]
                            
                            if tname in protected_tools and not is_client_authenticated(client_id):
                                output_content = "Authentication required. User must authenticate first before accessing this service."
                                messages.append({
                                    "role": "tool",
                                    "tool_call_id": getattr(tool_call, "id", None),
                                    "content": str(output_content)
                                })
                                continue
                            
                            if tname == "check_balance":
                                auth_details = get_client_auth_details(client_id)
                                api_result = await fetch_balance_from_api(
                                    last4=auth_details.get("last4"),
                                    mobile=auth_details.get("phone"),
                                    dob=auth_details.get("dob"),
                                    mpin=auth_details.get("mpin")
                                )
                                output_content = str(api_result)
                                
                            elif tname == "create_transaction":
                                trace_id = str(uuid.uuid4())
                                api_result = await create_transaction_in_api(bank_name=(args or {}).get("bank_name"), customer_name=(args or {}).get("customer_name"), customer_account_number=(args or {}).get("customer_account_number"), sender_account_number=(args or {}).get("sender_account_number"), amount=(args or {}).get("amount"), trace_id=trace_id)
                                output_content = api_result.get("speak")
                                    
                            elif tname == "get_last_transactions":
                                trace_id = str(uuid.uuid4())
                                limit = int((args or {}).get("limit") or 5)
                                
                                # ========== SECURITY CHECK: Strict Ownership Check ==========
                                req_acct = str((args or {}).get("customer_account_number", ""))
                                auth_details = get_client_auth_details(client_id)
                                stored_last4 = str(auth_details.get("last4", ""))
                                
                                if not req_acct.endswith(stored_last4):
                                    output_content = "I cannot access the requested information. You can only view transactions for your own authenticated account."
                                    log.warning(f"SECURITY ALERT: Unauthorized access attempt for account {req_acct}")
                                else:
                                    api_result = await fetch_last_transactions_from_api(customer_account_number=req_acct, limit=limit, trace_id=trace_id)
                                    output_content = json.dumps(api_result, default=str)
                                    txs = api_result.get("transactions") or []
                                    last_tx_cache_account = req_acct
                                    last_tx_cache_transactions = txs
                                    last_tx_cache_limit = limit
                                    last_tx_cache_time = datetime.now()

                            elif tname == "block_card":
                                card_number = (args or {}).get("card_number")
                                card_type = (args or {}).get("card_type")
                                confirm = (args or {}).get("confirm", False)
                                
                                if not card_number or not card_type:
                                    output_content = "Please provide the last 4-digit card number and card type."
                                elif not confirm:
                                    output_content = "Confirm: Are you sure you want to block this card?"
                                else:
                                    digits = ''.join(filter(str.isdigit, str(card_number or "")))
                                    trace = str(uuid.uuid4())
                                    res = await block_card_in_api(card_number=digits, card_type=card_type, trace_id=trace)
                                    output_content = res.get("speak", "Card blocking failed.")

                            else:
                                output_content = f"Tool {tname} not implemented."

                            messages.append({
                                "role": "tool", 
                                "tool_call_id": getattr(tool_call, "id", None), 
                                "content": str(output_content)
                            })

                        except Exception:
                            messages.append({
                                "role": "tool", 
                                "tool_call_id": getattr(tool_call, "id", None), 
                                "content": "Error executing tool"
                            })

                    if skip_post_tool_llm:
                        continue

                    try:
                        stream = await client.chat.completions.create(model=OPENAI_MODEL, messages=_merge_system_messages(messages), stream=True)
                        full_content = ""
                        async for chunk in stream:
                            delta = getattr(chunk.choices[0], "delta", None)
                            text_chunk = getattr(delta, "content", None) if delta else None
                            if text_chunk:
                                full_content += text_chunk
                                try:
                                    cleaned_chunk = _strip_markup_inline(text_chunk)
                                except Exception:
                                    cleaned_chunk = text_chunk
                                await websocket.send_json({"type": "bot_chunk", "text": cleaned_chunk})
                        final_reply = add_empathy_prefix(user_text, full_content)
                        messages.append({"role": "assistant", "content": final_reply})
                        await websocket.send_json({"type": "bot_done"})
                    except Exception:
                        log.exception("Streaming response generation failed")
                        await websocket.send_json({"type": "bot_chunk", "text": "System error generating response."})
                        await websocket.send_json({"type": "bot_done"})
                    continue

                else:
                    bot_text = getattr(response_msg, "content", None) if response_msg else None
                    if bot_text:
                        bot_text = _strip_markup_inline(bot_text)
                        messages.append({"role": "assistant", "content": bot_text})
                        if not client_in_human_mode(client_id):
                            bot_text = add_empathy_prefix(user_text, bot_text)
                            await websocket.send_json({"type": "bot_chunk", "text": bot_text})
                            await websocket.send_json({"type": "bot_done"})

            except WebSocketDisconnect:
                log.info("Chat Client Disconnected")
                break
            except Exception:
                log.exception("Unhandled error while processing websocket message")
                try:
                    await websocket.send_json({"type": "bot_chunk", "text": "Internal server error."})
                    await websocket.send_json({"type": "bot_done"})
                except Exception:
                    pass
                continue

    except Exception:
        log.exception("Critical websocket handler error")
    finally:
        if client_id in active_customers:
            del active_customers[client_id]
        if client_id in pending_transfers:
            try:
                pending_transfers.remove(client_id)
            except ValueError:
                pass
        if client_id in client_histories:
            del client_histories[client_id]
        if client_id in authenticated_clients:
            del authenticated_clients[client_id]
        if client_id in pending_protected_operations:
            del pending_protected_operations[client_id]
        lead_id = None
        if client_id in client_to_lead:
            lead_id = client_to_lead.pop(client_id, None)
            if lead_id and client_lead_map.get(lead_id) == client_id:
                client_lead_map.pop(lead_id, None)


# --- AGENT CHAT ENDPOINT (VICIDIAL) ---
@app.websocket("/ws/agent_chat/{lead_id}")
async def agent_chat_endpoint(websocket: WebSocket, lead_id: str):
    await websocket.accept()
    lead_id = str(lead_id).strip()
    log.info(f"?DEBUG: Agent attempting connection with Lead ID: '{lead_id}'")
    if pending_transfers:
        waiting_client_id = pending_transfers.pop(0)
        client_lead_map[lead_id] = waiting_client_id
        client_to_lead[waiting_client_id] = lead_id
        log.info(f"AUTO-MATCH SUCCESS: Linked Agent Lead {lead_id} <--> Client {waiting_client_id}")
        history_text = client_histories.get(waiting_client_id)
        if history_text:
            try:
                await websocket.send_json({"type": "bot_chunk", "text": history_text})
                await websocket.send_json({"type": "bot_done"})
                log.info(f"Chat history sent to Agent {lead_id}")
            except Exception as e:
                log.error(f"Failed to send history to agent: {e}")
    else:
        log.warning(f"Agent {lead_id} connected, but no clients are waiting in pending_transfers.")
    log.info(f"Current client_lead_map keys: {list(client_lead_map.keys())}")
    active_agents[lead_id] = websocket
    log.info(f"?Agent connected for Lead {lead_id}")
    try:
        while True:
            data = await websocket.receive_json()
            agent_text = data.get("text")
            target_client_id = client_lead_map.get(lead_id)
            if target_client_id and target_client_id in active_customers:
                customer_ws = active_customers[target_client_id]
                try:
                    if agent_text:
                        await customer_ws.send_json({"type": "bot_chunk", "text": agent_text})
                        await customer_ws.send_json({"type": "bot_done"})
                    log.info(f"Relayed Agent message to Client {target_client_id}")
                except Exception as e:
                    log.error(f"Failed to send to customer: {e}")
            else:
                log.warning(f"No active customer found for Lead {lead_id}")
    except WebSocketDisconnect:
        log.info(f"Agent {lead_id} disconnected")
        if lead_id in active_agents:
            del active_agents[lead_id]
        client_id = client_lead_map.pop(lead_id, None)
        if client_id:
            if client_to_lead.get(client_id) == lead_id:
                client_to_lead.pop(client_id, None)
        log.info(f"Cleaned mapping for Agent {lead_id} -> Client {client_id}")
    except Exception as e:
        log.error(f"Agent Chat Error: {e}")


@app.post("/api/validate-cheque")
async def validate_cheque_endpoint(file: UploadFile = File(...)):
    try:
        allowed_extensions = ['.png', '.jpg', '.jpeg', '.pdf']
        if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
            raise HTTPException(status_code=400, detail="Only PNG, JPG, JPEG, or PDF files are allowed")
        file.file.seek(0, 2)
        file_size = file.file.tell() / (1024 * 1024)
        file_size = file.file.tell() / (1024 * 1024)
        file.file.seek(0)
        if file_size > 10:
            raise HTTPException(status_code=400, detail="File size exceeds 10MB limit")
        result = await validate_cheque_image(file)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        log.error(f"Cheque validation endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/cheque-history")
async def get_cheque_history():
    try:
        cheques = []
        for filename in os.listdir(CHEQUE_DIR):
            if any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.pdf']):
                file_path = os.path.join(CHEQUE_DIR, filename)
                stats = os.stat(file_path)
                cheques.append({"filename": filename, "uploaded_at": datetime.fromtimestamp(stats.st_ctime).isoformat(), "size_kb": round(stats.st_size / 1024, 2)})
        return {"cheques": sorted(cheques, key=lambda x: x["uploaded_at"], reverse=True)}
    except Exception as e:
        log.error(f"Error getting cheque history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/start-kyc")
async def start_kyc_session():
    try:
        session_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        return {"success": True, "session_id": session_id, "timestamp": timestamp, "duration_seconds": KYC_DURATION_SECONDS, "message": "KYC session started. Please record 15-second video."}
    except Exception as e:
        log.error(f"Start KYC error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/submit-kyc")
async def submit_kyc_video(session_id: str = Form(...), user_id: str = Form(...), video: UploadFile = File(...), national_id: UploadFile = File(None), passport: UploadFile = File(None), address_proof: UploadFile = File(None)):
    try:
         # ADD THESE DEBUG LINES
        print(f"\nKYC SUBMISSION RECEIVED")
        print(f"    user_id: {user_id}")
        print(f"    account_intake_by_client keys: {list(account_intake_by_client.keys())}")

        # FIX: Try to find matching client_id in different formats
        entered = {}

         # First, try direct match
        if user_id in account_intake_by_client:
            entered = account_intake_by_client.get(user_id, {})
            print(f"Direct match found for: {user_id}")
        
        # If not found and user_id starts with 'client_', try to extract and find UUID
        elif user_id.startswith('client_'):
            # Look for any UUID key in account_intake_by_client
            for key in account_intake_by_client.keys():
                # Check if this is a UUID format key
                if len(key) == 36 and '-' in key:  # UUID format
                    print(f"    Attempting to match client_ format with UUID: {key}")
                    entered = account_intake_by_client.get(key, {})
                    if entered:
                        print(f"Found matching UUID client: {key}")
                        break
        
        # If still not found, try the reverse - maybe the key is stored with client_ prefix
        else:
            client_prefix_key = f"client_{user_id}" if not user_id.startswith('client_') else user_id
            if client_prefix_key in account_intake_by_client:
                entered = account_intake_by_client.get(client_prefix_key, {})
                print(f"Found with client_ prefix: {client_prefix_key}")
        
        print(f"    Found details: {bool(entered)}")
        if entered:
            print(f"    Details: {entered}")
        else:
            print(f"No details found for user_id: {user_id}")


        # All three documents are required (ID card, PAN card / passport, proof of address).
        missing_docs = [label for label, f in (("ID card (Aadhaar / National ID)", national_id),
                                               ("PAN card or Passport", passport),
                                               ("proof of address", address_proof)) if f is None]
        if missing_docs:
            message = "Please upload all three documents. Missing: " + ", ".join(missing_docs) + "."
            return JSONResponse(status_code=400, content={
                "success": False, "message": message,
                "verification": {"status": "Failed", "message": message, "face_match": False, "face_confidence": 0},
            })

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        doc_paths = {}
        async def save_file(upload_file, label):
            if upload_file:
                ext = upload_file.filename.split(".")[-1]
                path = os.path.join(KYC_VIDEO_DIR, f"{label}_{user_id}_{timestamp}.{ext}")
                content = await upload_file.read()
                with open(path, "wb") as f:
                    f.write(content)
                return path
            return None
            
        doc_paths["national_id"] = await save_file(national_id, "id")
        doc_paths["passport"] = await save_file(passport, "pass")
        doc_paths["address_proof"] = await save_file(address_proof, "addr")
        raw_video_path = await save_file(video, "raw_video")
        meta, frames_list = await save_kyc_video(raw_video_path, user_id, session_id, doc_paths)
        
        if not frames_list:
            return JSONResponse(status_code=400, content={"success": False, "message": "Video processing failed: No frames could be read from the video."})
        
        mid_idx = len(frames_list) // 2
        frame_bytes = decode_frame_any_format(frames_list[mid_idx])
        best_frame_path = None
        if frames_list and len(frames_list) > 0:
            best_frame_path = os.path.join(KYC_VIDEO_DIR, f"face_ref_{user_id}_{timestamp}.jpg")
            frame_data = decode_frame_any_format(frames_list[len(frames_list)//2])
            with open(best_frame_path, "wb") as f:
                f.write(frame_data)
        
        log.info(f"DEBUG: ID Path: {doc_paths.get('national_id')}")
        log.info(f"DEBUG: Video Frame Path: {best_frame_path}")
        
        verification_report = verify_documents(
            national_id_path=doc_paths.get("national_id"), 
            passport_path=doc_paths.get("passport"), 
            address_proof_path=doc_paths.get("address_proof"), 
            video_frame_path=best_frame_path
        )
        
        # FIX: Get entered details from account_intake_by_client
        entered = account_intake_by_client.get(user_id, {})
        print(f"    Found details: {bool(entered)}")
        if entered:
            print(f"    Details: {entered}")
        else:
            print(f"WARNING: No details found for user_id: {user_id}")
        
        # Compare entered name with extracted name and add match status
        entered_name = f"{entered.get('first_name', '')} {entered.get('last_name', '')}".strip()
        extracted_name = verification_report.get("person_data", {}).get("name", "Not Detected")
        
        # The name may be on any line of the ID or passport (a guessed line can be OCR noise, e.g. a
        # passport card): accept a line that contains every word of the name the customer typed.
        ocr_texts = verification_report.pop("_ocr", {}) or {}
        name_ok = names_match(entered_name, extracted_name)
        if not name_ok and entered_name:
            for text in ocr_texts.values():
                line = next((ln.strip() for ln in (text or "").splitlines() if names_match(entered_name, ln)), None)
                if line:
                    extracted_name, name_ok = re.sub(r"\s+", " ", line), True
                    verification_report.setdefault("person_data", {})["name"] = extracted_name
                    break
        verification_report["name_match"] = {"entered": entered_name, "extracted": extracted_name, "matched": name_ok}
        liveness = meta.get("metrics", {}) if isinstance(meta, dict) else {}
        verification_report["liveness"] = liveness
        verification_report["address_data"] = check_address_proof(doc_paths.get("address_proof"), extracted_name, entered)

        # Verified only when EVERY check passes; otherwise say plainly what failed.
        reasons = []
        if not entered:
            reasons.append("we could not find the details you entered in the chat, please start again from the chat")
        if not verification_report.get("face_match"):
            reasons.append("your face could not be matched with the photo on your ID card"
                           if verification_report.get("face_confidence") else
                           "we could not see a face clearly in the video or on the ID card")
        if verification_report["id_data"]["number"] == "Not Detected":
            reasons.append("we could not read the number on your ID card")
        if verification_report["passport_data"]["number"] == "Not Detected":
            reasons.append("we could not read the number on your PAN card / passport")
        if entered and not name_ok:
            reasons.append(f"the name you entered ({entered_name}) does not match the name on your ID ({extracted_name})")
        if not liveness.get("is_valid"):
            reasons.append("the face video check failed: " + "; ".join(liveness.get("problems") or ["no usable video"]))
        if verification_report["address_data"].get("reason"):
            reasons.append(verification_report["address_data"]["reason"])
        # The same file in two slots is a customer mistake worth explaining plainly.
        hashes = {}
        for slot, label in (("national_id", "ID card"), ("passport", "PAN card / passport"), ("address_proof", "proof of address")):
            path = doc_paths.get(slot)
            if path and os.path.exists(path):
                with open(path, "rb") as fh:
                    hashes.setdefault(hashlib.sha256(fh.read()).hexdigest(), []).append(label)
        for labels in hashes.values():
            if len(labels) > 1:
                reasons.insert(0, f"the same picture was uploaded as {' and '.join(labels)}; please upload each document separately")
        verification_report["reasons"] = reasons
        verification_report["status"] = "Verified" if not reasons else "Failed"
        verification_report["message"] = ("Identity verified successfully." if not reasons else
                                          "KYC could not be completed: " + "; ".join(reasons) + ".")
        log.info(f"KYC result for {user_id}: {verification_report['status']} ({len(reasons)} problem(s))")

        return JSONResponse({
            "success": True,
            "message": verification_report["message"],
            "verification": verification_report,
            "entered_details": entered,  # This is already here, but make sure it's populated
            "metadata": meta
        })
    except Exception as e:
        log.error(f"API Error: {str(e)}")
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


@app.get("/api/kyc-history")
async def get_kyc_history(user_id: str = None):
    try:
        kyc_records = []
        for filename in os.listdir(KYC_VIDEO_DIR):
            if filename.endswith(".json"):
                metadata_path = os.path.join(KYC_VIDEO_DIR, filename)
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                if user_id and metadata.get("user_id") != user_id:
                    continue
                kyc_records.append({"kyc_id": metadata.get("filename", "").replace(".mp4", ""), "user_id": metadata.get("user_id", "unknown"), "timestamp": metadata.get("timestamp"), "duration": metadata.get("duration_seconds", 0), "is_valid": metadata.get("validation_results", {}).get("is_valid", False), "validation_score": metadata.get("validation_results", {}).get("stability_score", 0), "file_path": metadata.get("file_path")})
        return {"kyc_records": sorted(kyc_records, key=lambda x: x["timestamp"], reverse=True)}
    except Exception as e:
        log.error(f"Error getting KYC history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")