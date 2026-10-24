import os
import json
import base64
import asyncio
import logging
import httpx
import uuid  # ✅ Added missing import
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from websockets.asyncio.client import connect
from dotenv import load_dotenv

# ✅ Knowledge Base Import
from knowledge_base import KB_TEXT

load_dotenv()
BANK_API_BASE_URL = os.getenv("BANK_API_BASE_URL", "http://localhost:9000/api").rstrip("/")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("❌ Error: OPENAI_API_KEY is missing in .env file.")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("VOICE")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPENAI_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-mini-realtime-preview"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "OpenAI-Beta": "realtime=v1",
}

# 🟢 SYSTEM INSTRUCTIONS (GLOBAL)
SYSTEM_INSTRUCTIONS = (
    "You are Demo Bank AI. "
    "Your goal is to assist users using the Knowledge Base below.\n\n"
    "==================== KNOWLEDGE BASE START ====================\n"
    f"{KB_TEXT}\n"
    "==================== KNOWLEDGE BASE END ======================\n\n"
    "RULES:\n"
    "0. **CRITICAL TABLE DISPLAY RULES:**\n"
    "1. When user asks about ANY of these topics, you MUST follow this flow:\n"
    "   a) Give a brief 1-2 sentence summary from the Knowledge Base\n"
    "   b) Then ALWAYS ask: 'Would you like this in table format?'\n"
    "   c) Wait for user response\n"
    "\n"
    "2. **TABLE TOPICS TRIGGER:**\n"
    "   - Credit cards, debit cards, card products, card eligibility, card fees\n"
    "   - Loans, personal loans, auto loans, mortgage, loan eligibility\n"
    "   - Bank accounts, account types, checking accounts, savings accounts\n"
    "\n"
    "3. **USER RESPONSE HANDLING:**\n"
    "   - If user says 'Yes' or agrees → send table using special format\n"
    "   - If user says 'No' → provide information in plain text\n"
    "   - If user asks something else → continue normal conversation\n"
    "\n"
    "**TABLE FORMAT (JSON):**\n"
    '{\n'
    '  "type": "table",\n'
    '  "title": "Title Here",\n'
    '  "columns": ["Col1", "Col2", ...],\n'
    '  "rows": [\n'
    '    ["Row1Col1", "Row1Col2", ...],\n'
    '    ["Row2Col1", "Row2Col2", ...]\n'
    '  ],\n'
    '  "note": "Additional notes if any"\n'
    '}\n'
    "\n"
    "Available tables:\n"
    "1. CREDIT CARDS:\n"
    "   - Card Eligibility Table (Card Type, Minimum Age, Minimum Income, Credit Score)\n"
    "   - Interest Rates Table (Card Type, Purchase APR, Cash Advance APR, Grace Period)\n"
    "   - Annual Fees Table (Card, Annual Fee, Waiver Conditions)\n"
    "   - Debit Card Fees Table (Transaction Type, Fee)\n"
    "\n"
    "2. LOANS:\n"
    "   - Loan options and eligibility\n"
    "   - Loan Types Comparison (Loan Type, Purpose, Term, Collateral)\n"
    "\n"
    "3. BANK ACCOUNTS:\n"
    "   - Account Types Comparison (Account Type, Purpose, Fees, Minimum Requirements)\n"
    "\n"
    "**EXTRACTION RULES:**\n"
    "1. Find data between <<<TABLE_CREDIT_CARDS_START>>> and <<<TABLE_CREDIT_CARDS_END>>>\n"
    "2. Find data between <<<TABLE_LOANS_START>>> and <<<TABLE_LOANS_END>>>\n"
    "3. Find data between <<<TABLE_ACCOUNTS_START>>> and <<<TABLE_ACCOUNTS_END>>>\n"
    "\n"
    
    "1. **STRICT LANGUAGE MATCHING (CRITICAL):** \n"
    "   - You MUST detect the language of the user's audio immediately.\n"
    "   - If the user speaks **English**, you MUST reply in **English**.\n"
    "   - If the user speaks **Hindi**, you MUST reply in **I Can Talk in English Only**.\n"
    "   - **NEVER** switch languages on your own. Always mirror the user.\n"
    "2. **EXTREME BREVITY (SAVE CREDITS):** \n"
    "   - Keep answers **very short** and **summarized**.\n"
    "   - Avoid polite filler phrases like 'Sure', 'I can help with that', or 'Please wait'.\n"
    "   - Get straight to the answer. Max 1-2 short sentences.\n"
    "3. PREFER the Knowledge Base for all answers.\n"
    "4. **NOISE & SILENCE HANDLING:** \n"
    "   - If the audio is just background noise or gibberish, **DO NOT RESPOND**.\n"
    "   - Do NOT say phrases like 'It seems there is noise'. Stay silent.\n"
    "5. **SECURITY PROTOCOL:** If the user asks for a Balance Enquiry, you MUST ask for these 3 details:\n"
    "   - Last 4 digits of Account Number\n"
    "   - Registered Mobile Number\n"
    "   - Date of Birth in DD-MM-YYYY format\n"
    "   **CRITICAL:** Once the user provides these 3 details, you MUST immediately use the `check_balance` tool. "
    "   Do NOT say 'Contact support'. Use the tool."
    "6. *SECURITY PROTOCOL – TRANSACTION:* Any transaction request is a privileged action.\n."
    "If the user requests a transaction and ANY of the following are missing, you MUST ask for them one by one:\n."
    "- bank_name\n."
    "- customer_account_number\n."
    "- customer_name\n."
    "- amount\n."
    " Do NOT guess, infer, or assume missing values.\n."
    "If multiple fields are provided together, store them and ask ONLY for the missing ones.\n."
    "Once all four fields are collected, you MUST immediately call the create_transaction tool.\n."
    "Do NOT ask any other questions during this workflow\n."
    "7. After a successful transaction tool response, confirm completion briefly and continue the conversation."
)

# ✅ TOOL DEFINITION
TOOLS = [
    {
        "type": "function",
        "name": "check_balance",
        "description": "Checks the account balance using the provided details.",
        "parameters": {
            "type": "object",
            "properties": {
                "last4": {"type": "string", "description": "Last 4 digits of account number"},
                "mobile_number": {"type": "string", "description": "Mobile number"},
                "dob": {"type": "string", "description": "Date of Birth (DD-MM-YYYY)"}
            },
            "required": ["last4", "mobile_number", "dob"]
        }
    },
    {
        "type": "function",
        "name": "create_transaction",
        "description": "Create a transaction record via the core-banking API",
        "parameters": {
            "type": "object",
            "properties": {
                "bank_name": {"type": "string", "description": "Bank name"},
                "customer_name": {"type": "string", "description": "Customer name"},
                "customer_account_number": {"type": "string", "description": "Customer account number"},
                "amount": {"type": "number", "description": "Amount to debit/credit"}
            },
            "required": ["bank_name", "customer_name", "customer_account_number", "amount"]
        }
    },
    {
        "type": "function",
        "name": "transfer_call",
        "description": "Transfers the user to a human agent. Use this ONLY when the user explicitly asks to speak to a person.",
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {"type": "string", "description": "Reason for transfer"}
            },
            "required": []
        }
    },
    {
        "type": "function",
        "name": "get_last_transactions",
        "description": "Fetch last N transactions for an account via the core-banking API",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_account_number": {"type": "string", "description": "Customer account number"},
                "limit": {"type": "integer", "description": "Number of transactions to fetch (default 5)"}
            },
            "required": ["customer_account_number"]
        }
    },
    # ---- ADDED: block_card tool ----
    {
        "type": "function",
        "name": "block_card",
        "description": "Block/deactivate a customer's card. Requires verification to be collected by the LLM first.",
        "parameters": {
            "type": "object",
            "properties": {
                "card_number": {"type": "string", "description": "12-digit card number (mandatory)"},
                "card_type": {"type": "string", "description": "Card type: debit/credit/mastercard"},
                "confirm": {"type": "boolean", "description": "Must be true to proceed"}
            },
            "required": ["card_number", "card_type"]
        }
    }
]

# ✅ VAD SETTINGS
VAD_PARAMS = {
    "type": "server_vad",
    "threshold": 0.6,
    "prefix_padding_ms": 300,
    "silence_duration_ms": 500
}

SESSION_UPDATE = {
    "type": "session.update",
    "session": {
        "modalities": ["text", "audio"],
        "voice": "alloy",
        "input_audio_format": "pcm16",
        "output_audio_format": "pcm16",
        "input_audio_transcription": {"model": "whisper-1"},
        "turn_detection": None,  # 🔴 Initially OFF (No interruption allowed during greeting)
        "instructions": SYSTEM_INSTRUCTIONS,
        "tools": TOOLS,
        "tool_choice": "auto",
        "temperature": 0.6,
    },
}

GREETING_TEXT = (
    "Hi this is Amelia, your friendly AI Assistant from Demo Bank. For quality & security purposes this conversation will be recorded. How can I help you today? "
)

# -------------------------------------------------------------------------
# ✅ API HELPER FUNCTIONS
# -------------------------------------------------------------------------

async def fetch_balance_from_api(last4, mobile, dob):
    url = f"{BANK_API_BASE_URL}/auth"
    payload = {
        "last4": last4,
        "system_caller_id": None,
        "system_called_number": mobile,
        "dob": dob
    }
    log.info(f"📡 Balance API Request: last4={last4}")
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=8.0)
            data = resp.json()
            log.info(f"✅ Balance API Response: {data}")

            speak = (data.get("speak") or "").strip()
            cust_name = (data.get("customer") or {}).get("name")

            if cust_name:
                speak = f"I see you as {cust_name}. {speak}"

            return speak or "I could not verify your details. Please try again."

        except Exception as e:
            log.error(f"❌ Balance API Error: {e}")
            return "System error while checking balance."

async def create_transaction_in_api(bank_name, customer_name, customer_account_number, amount, trace_id=None):
    log.critical("🔥 create_transaction_in_api() CALLED 🔥")
    if not trace_id:
        trace_id = str(uuid.uuid4())

    url = f"{BANK_API_BASE_URL}/transfer"
    payload = {
        "bank_name": bank_name,
        "customer_name": customer_name,
        "customer_account_number": customer_account_number,
        "amount": amount,
        "trace_id": trace_id
    }

    log.info(f"📡 Transaction API Request ({trace_id}): {payload}")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=10.0)
            log.info(f"HTTP {resp.status_code} FROM transaction API: {resp.text}")
            try:
                data = resp.json()
            except Exception as e:
                log.error(f"❌ JSON parse error from transaction API: {e} - body: {resp.text}")
                data = {"success": False, "message": "Invalid JSON from backend", "raw": resp.text}

            log.info(f"✅ Transaction API Response ({trace_id}): {data}")

            if data.get("success"):
                tx_id = data.get("transaction_id")
                return {
                    "speak": f"Transaction recorded successfully. Transaction ID {tx_id}.",
                    "data": data,
                    "trace_id": trace_id
                }
            else:
                return {
                    "speak": data.get("message") or data.get("error") or "Transaction failed due to server error.",
                    "data": data,
                    "trace_id": trace_id
                }
        except Exception as e:
            log.error(f"❌ Transaction API Error ({trace_id}): {e}")
            return {"speak": "System error while creating transaction.", "data": {}, "trace_id": trace_id}

async def fetch_last_transactions_from_api(customer_account_number, limit=5, trace_id=None):
    log.critical("🔥 fetch_last_transactions_from_api() CALLED 🔥")
    if not trace_id:
        trace_id = str(uuid.uuid4())

    url = f"{BANK_API_BASE_URL}/transactions"
    payload = {
        "customer_account_number": customer_account_number,
        "limit": limit,
        "trace_id": trace_id
    }

    log.info(f"📡 GetLastTx API Request ({trace_id}): {payload}")

    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=8.0)
            log.info(f"HTTP {resp.status_code} FROM GetLastTx API: {resp.text}")
            try:
                data = resp.json()
            except Exception as e:
                log.error(f"❌ JSON parse error from GetLastTx API: {e} - body: {resp.text}")
                data = {"success": False, "message": "Invalid JSON from backend", "raw": resp.text}

            log.info(f"✅ GetLastTx API Response ({trace_id}): {data}")

            if data.get("success") and isinstance(data.get("last_transactions"), list):
                return {
                    "speak": "Fetched last transactions.", 
                    "transactions": data.get("last_transactions"), 
                    "data": data, 
                    "trace_id": trace_id
                }
            else:
                return {
                    "speak": data.get("message") or data.get("error") or "No transactions found.", 
                    "transactions": [], 
                    "data": data, 
                    "trace_id": trace_id
                }
        except Exception as e:
            log.error(f"❌ GetLastTx API Error ({trace_id}): {e}")
            return {"speak": "System error while fetching transactions.", "transactions": [], "data": {}, "trace_id": trace_id}

# ---- ADDED: Block card helper ----
async def block_card_in_api( card_number, card_type, trace_id=None):
    """
    Calls backend PHP API to deactivate a card (table 'cards', column 'card_id').
    Expects backend to accept JSON: card_id, mobile_number, last4, dob, card_type, trace_id
    """
    log.critical("🔥 block_card_in_api() CALLED 🔥")
    if not trace_id:
        trace_id = str(uuid.uuid4())

    # Validate card_number is 12 digits before calling
    card_clean = ''.join(filter(str.isdigit, str(card_number)))
    if len(card_clean) != 12:
        return {"speak": "Card number must be exactly 12 digits.", "success": False, "trace_id": trace_id}

    url = f"{BANK_API_BASE_URL}/block-card"
    payload = {
        "card_id": card_clean,
        "card_type": card_type,
        "trace_id": trace_id,
        "action": "BLOCKED"
    }

    log.info(f"📡 BlockCard API Request ({trace_id}): {payload}")
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.post(url, json=payload, timeout=10.0)
            log.info(f"HTTP {resp.status_code} FROM BlockCard API: {resp.text}")
            try:
                data = resp.json()
            except Exception as e:
                log.error(f"❌ JSON parse error from BlockCard API: {e} - body: {resp.text}")
                data = {"success": False, "message": "Invalid JSON from backend", "raw": resp.text}

            log.info(f"✅ BlockCard API Response ({trace_id}): {data}")

            if data.get("success"):
                return {"speak": f"Your {card_type} has been blocked successfully.", "success": True, "data": data, "trace_id": trace_id}
            else:
                return {"speak": data.get("message") or data.get("error") or "Failed to block the card.", "success": False, "data": data, "trace_id": trace_id}

        except Exception as e:
            log.error(f"❌ BlockCard API Error ({trace_id}): {e}")
            return {"speak": "System error while blocking the card.", "success": False, "data": {}, "trace_id": trace_id}
        
#ayush
def _fmt_money_usd(x):
    try:
        return f"${float(x):,.2f}"
    except Exception:
        return str(x or "")

def _safe(v):
    return "" if v is None else str(v)

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
    # fallback if user says "No"
    lines = []
    for tx in (transactions or []):
        date = _safe(tx.get("Date") or tx.get("date") or tx.get("txn_date"))
        typ = _safe(tx.get("Type") or tx.get("type"))
        amt = tx.get("Amount") or tx.get("amount") or ""
        lines.append(f"{date} — {typ} — {_fmt_money_usd(amt)}")
    return "\n".join(lines) if lines else "No transactions found."


def _kb_sections(kb_text: str) -> dict:
    """Split the knowledge base into its '## Heading' sections."""
    sections, current = {}, None
    for line in kb_text.splitlines():
        if line.startswith("## "):
            current = line[3:].strip().lower()
            sections[current] = []
        elif current:
            sections[current].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


TOPIC_SECTIONS = {
    "card": "cards",
    "loan": "loans",
    "account": "accounts",
    "deposit": "fixed deposits",
}


def _section_for_topic(topic: str, kb_text: str) -> tuple[str, str]:
    topic_lower = topic.lower()
    sections = _kb_sections(kb_text)
    for word, name in TOPIC_SECTIONS.items():
        if word in topic_lower and name in sections:
            return name, sections[name]
    return "", ""


def get_table_for_topic(topic: str, kb_text: str):
    """Build a two-column table (item, details) from the KB section for the topic."""
    name, body = _section_for_topic(topic, kb_text)
    if not body:
        return None
    rows = []
    for line in body.splitlines():
        line = line.lstrip("- ").strip()
        if ":" in line:
            item, details = line.split(":", 1)
            rows.append([item.strip(), details.strip()])
    if not rows:
        return None
    return {"title": f"Demo Bank {name.title()}", "columns": ["Item", "Details"], "rows": rows}


def extract_text_for_topic(topic: str, kb_text: str) -> str:
    """Return the KB section for a topic when the user doesn't want a table."""
    _, body = _section_for_topic(topic, kb_text)
    return body or "I don't have details on that yet. Would you like me to connect you to an agent?"





# -------------------------------------------------------------------------
# ✅ WEBSOCKET ENDPOINT
# -------------------------------------------------------------------------

@app.websocket("/ws/voice")
async def voice_ws(client_ws: WebSocket):
    await client_ws.accept()
    log.info("🔗 Client connected")

    greeting_completed = False
    
    # Store mapping of call_id -> function_name
    function_call_map = {}
    #ayush
    # ---- Tx table confirmation state (VOICE) ----
    pending_tx_table_choice = False
    pending_tx_transactions = None
    pending_tx_limit = 5

    pending_table_topic = None
    pending_table_data = None
    

    
    


    # Send text greeting to UI immediately
    await client_ws.send_json({
        "type": "bot_text",
        "text": GREETING_TEXT
    })

    try:
        async with connect(OPENAI_URL, additional_headers=HEADERS) as openai_ws:
            log.info("🤖 Connected to OpenAI Realtime")

            # Initialize session (VAD IS OFF HERE)
            await openai_ws.send(json.dumps(SESSION_UPDATE))

            # Speak greeting
            await asyncio.sleep(0.1)
            await openai_ws.send(json.dumps({
                "type": "response.create",
                "response": {
                    "modalities": ["text", "audio"],
                    "instructions": f"Say exactly this greeting: {GREETING_TEXT}"
                }
            }))

            async def client_to_openai():
                try:
                    while True:
                        msg = await client_ws.receive()
                        

                        # 1) AUDIO from client -> OpenAI Realtime
                        if msg["type"] == "websocket.receive" and "bytes" in msg and msg["bytes"]:
                            audio_b64 = base64.b64encode(msg["bytes"]).decode()
                            await openai_ws.send(json.dumps({
                                "type": "input_audio_buffer.append",
                                "audio": audio_b64
                            }))
                        #ayush
                        # 2) TEXT from client (Yes/No/account number) -> OpenAI Realtime
                        elif msg["type"] == "websocket.receive" and "text" in msg and msg["text"]:
                            raw = msg["text"]

                            # Accept either plain text "Yes" OR JSON like {"text":"Yes"}
                            try:
                                payload = json.loads(raw)
                                user_text = (payload.get("text") or "").strip()
                            except Exception:
                                user_text = raw.strip()

                            nonlocal pending_table_topic, pending_table_data
                            nonlocal pending_tx_table_choice, pending_tx_transactions, pending_tx_limit

                            await client_ws.send_json({
                                "type": "user_text",
                                "text": user_text
                            });  

                            

                            # Check if this is a response to table question
                            if pending_table_data and user_text:
                                user_lower = user_text.lower()
                                
                                if any(word in user_lower for word in ["yes", "yeah", "yep", "sure", "ok", "okay", "please"]):
                                    # User wants table - send it
                                    log.info(f"📊 User requested table for: {pending_table_topic}")
                                    
                                    # Send table to frontend
                                    await client_ws.send_json({
                                        "type": "table",
                                        "title": pending_table_data.get("title", pending_table_topic),
                                        "columns": pending_table_data.get("columns", []),
                                        "rows": pending_table_data.get("rows", []),
                                        "note": pending_table_data.get("note", "")
                                    })
                                    
                                    # Also speak confirmation
                                    await openai_ws.send(json.dumps({
                                        "type": "response.create",
                                        "response": {
                                            "modalities": ["text", "audio"],
                                            "instructions": "Here's the information in table format."
                                        }
                                    }))
                                    
                                elif any(word in user_lower for word in ["no", "nope", "nah", "text"]):
                                    # User wants text - extract from KB
                                    log.info(f"📝 User wants text format for: {pending_table_topic}")
                                    
                                    # Find relevant text from KB
                                    kb_section = extract_text_for_topic(pending_table_topic, KB_TEXT)
                                    if kb_section:
                                        await client_ws.send_json({
                                            "type": "bot_text",
                                            "text": kb_section[:500]  # Limit length
                                        })
                                    
                                # Reset table state
                                pending_table_topic = None
                                pending_table_data = None
                                
                                # Don't send to OpenAI if it was just a table response
                                continue 



                            

                            if user_text:
                                # Put user's text into the realtime conversation
                                await openai_ws.send(json.dumps({
                                    "type": "conversation.item.create",
                                    "item": {
                                        "type": "message",
                                        "role": "user",
                                        "content": [{"type": "input_text", "text": user_text}]
                                    }
                                }))

                                # IMPORTANT: trigger a response
                                await openai_ws.send(json.dumps({"type": "response.create"}))

                        # 3) Disconnect
                        elif msg["type"] == "websocket.disconnect":
                            raise WebSocketDisconnect

                except WebSocketDisconnect:
                    log.info("Client disconnected")


            async def openai_to_client():
                nonlocal greeting_completed, pending_table_topic, pending_table_data
                nonlocal pending_tx_table_choice, pending_tx_transactions, pending_tx_limit 
                async for message in openai_ws:
                    data = json.loads(message)
                    event_type = data.get("type")

                    # 1. Stream Audio
                    if event_type == "response.audio.delta":
                        await client_ws.send_bytes(base64.b64decode(data["delta"]))

                    # 2. Handle Interruption
                    if event_type == "input_audio_buffer.speech_started":
                        await client_ws.send_json({"type": "user_start"})
                        await client_ws.send_json({"type": "interrupt"})
                        await openai_ws.send(json.dumps({"type": "response.cancel"}))

                    # 3. User Transcript
                                       # 3. User Transcript
                    if event_type == "conversation.item.input_audio_transcription.completed":
                        transcript = (data.get("transcript") or "").strip()

                        # 🎤 CRITICAL DEBUG LOG
                        log.critical(f"🎤🎤🎤 TRANSCRIPT RECEIVED FROM OPENAI: '{transcript}'")

                        if not transcript:
                            await client_ws.send_json({"type": "user_empty"})
                            continue


                        

                        # ✅ 0) If waiting for tx table confirmation, handle it FIRST    #ayush
                        nonlocal pending_tx_table_choice, pending_tx_transactions, pending_tx_limit

                        if pending_tx_table_choice and pending_tx_transactions is not None:
                            u = transcript.lower().strip()
                            yes_words = {"yes", "y", "yeah", "yep", "ok", "okay", "sure", "haan", "ha"}
                            no_words  = {"no", "n", "nope", "nah", "na"}

                            # Stop any auto-response
                            await openai_ws.send(json.dumps({"type": "response.cancel"}))

                            if (u in yes_words) or ("yes" in u) or ("table" in u):
                                # 1) send table to UI
                                await client_ws.send_json(
                                    build_transactions_table_payload(
                                        pending_tx_transactions,
                                        title=f"Last {pending_tx_limit} transactions"
                                    )
                                )

                                speak = f"Showing your last {pending_tx_limit} transactions in a table."
                                await client_ws.send_json({"type": "bot_text", "text": speak})

                                await openai_ws.send(json.dumps({
                                    "type": "response.create",
                                    "response": {"modalities": ["text", "audio"], "instructions": f"Say exactly this: {speak}"}
                                }))

                            elif (u in no_words) or ("no" in u):
                                text = build_transactions_text(pending_tx_transactions)
                                await client_ws.send_json({"type": "bot_text", "text": text})

                                await openai_ws.send(json.dumps({
                                    "type": "response.create",
                                    "response": {"modalities": ["text", "audio"], "instructions": f"Say exactly this: {text}"}
                                }))

                            else:
                                retry = "Would you like this in table format? Please say Yes or No."
                                await client_ws.send_json({"type": "bot_text", "text": retry})

                                await openai_ws.send(json.dumps({
                                    "type": "response.create",
                                    "response": {"modalities": ["text", "audio"], "instructions": f"Say exactly this: {retry}"}
                                }))
                                

                            # reset state and stop further processing for this turn
                            pending_tx_table_choice = False
                            pending_tx_transactions = None
                            pending_tx_limit = 5
                            continue

                        # ✅ FIX: Add this AFTER the table handling
                        # Send user's speech to frontend
                        log.critical(f"🎤 SENDING user_text to frontend: '{transcript}'")
                        await client_ws.send_json({
                            "type": "user_text",
                            "text": transcript
                        })

                         # Also add the user's message to the OpenAI conversation
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "message",
                                "role": "user",
                                "content": [{"type": "input_text", "text": transcript}]
                            }
                        }))

                         # Trigger AI response
                        await openai_ws.send(json.dumps({"type": "response.create"}))
                        



                        # HARD TRANSFER
                        t = transcript.lower()

                        if "transfer" in t and ("agent" in t or "human" in t or "person" in t):
                            log.critical("🔥 DIRECT TRANSFER TRIGGERED (bypassing AI)")

                            await openai_ws.send(json.dumps({"type": "response.cancel"}))

                            await client_ws.send_json({
                                "type": "frontend_action",
                                "action": "call_agent",
                                "destination": "1001"
                            })

                            await client_ws.send_json({
                                "type": "transfer",
                                "target": "1001"
                            })

                            # Still show what user said in chat
                            await client_ws.send_json({
                                "type": "user_text",
                                "text": transcript
                            })

                            # ✅ THEN handle normal user speech
                            # Send user's speech to frontend
                            log.critical(f"🎤🎤🎤 SENDING user_text to frontend: '{transcript}'")
                            await client_ws.send_json({
                                "type": "user_text",
                                "text": transcript
                            })

                            # # Also add the user's message to the OpenAI conversation
                            # log.critical(f"🎤🎤🎤 Adding to OpenAI conversation: '{transcript}'")
                            # await openai_ws.send(json.dumps({
                            #     "type": "conversation.item.create",
                            #     "item": {
                            #         "type": "message",
                            #         "role": "user",
                            #         "content": [{"type": "input_text", "text": transcript}]
                            #     }
                            # }))
                            
                            # # Trigger AI response
                            # log.critical("🎤🎤🎤 Triggering AI response")
                            # await openai_ws.send(json.dumps({"type": "response.create"}))
                            log.critical("🔵 NOT sending to OpenAI (already has audio)")
                            continue


                    # 4. Bot Transcript
                    if event_type == "response.audio_transcript.done":
                        transcript = (data.get("transcript") or "").strip()
                        if transcript:
                            # Check if this is a topic that has table data
                            topic_keywords = {
                                "card": ["card", "credit", "debit", "visa", "mastercard"],
                                "loan": ["loan", "mortgage", "finance", "borrow", "emi"],
                                "account": ["account", "checking", "savings", "deposit", "bank account"]
                            }
                            
                            transcript_lower = transcript.lower()
                            
                            # Check for table topics in user query
                            for category, keywords in topic_keywords.items():
                                if any(keyword in transcript_lower for keyword in keywords):
                                    # User is asking about this topic
                                    table_data = get_table_for_topic(category, KB_TEXT)
                                    if table_data:
                                        pending_table_topic = category
                                        pending_table_data = table_data
                                        log.info(f"📋 Detected {category} topic for table display")
                                        
                                        # Ask if user wants table format
                                        await client_ws.send_json({
                                            "type": "bot_text",
                                            "text": "Would you like this in table format?"
                                        })
                                        
                                        # Also send frontend action for quick replies
                                        await client_ws.send_json({
                                            "type": "frontend_action",
                                            "action": "show_quick_replies",
                                            "options": ["Yes", "No"]
                                        })
                                    break

                            # 🟢 Enable VAD after greeting is completed
                            if not greeting_completed and "Demo Bank" in transcript:
                                greeting_completed = True
                                log.info("✅ Greeting finished. Enabling Interruption Detection (VAD) now.")
                                await openai_ws.send(json.dumps({
                                    "type": "session.update",
                                    "session": {
                                        "turn_detection": VAD_PARAMS
                                    }
                                }))
                                continue 
                            
                            
                            await client_ws.send_json({
                                "type": "bot_text",
                                "text": transcript
                            })

                    # 5. Store Function Name when item is added
                    if event_type == "response.output_item.added":
                        item = data.get("item", {})
                        if item.get("type") == "function_call":
                            call_id = item.get("call_id")
                            func_name = item.get("name")
                            if call_id and func_name:
                                function_call_map[call_id] = func_name
                                log.info(f"📝 Registered function call: {call_id} -> {func_name}")

                    # 6. HANDLE TOOL CALL (Arguments Done)
                    if event_type == "response.function_call_arguments.done":
                        call_id = data.get("call_id")
                        args_str = data.get("arguments", "{}")
                        args = json.loads(args_str)
                        
                        # Identify which function was called
                        func_name = function_call_map.get(call_id)
                        log.info(f"🛠️ Executing Tool: {func_name} with args: {args}")

                        output_speak = "System error."
                        
                        # --- Dispatch Logic ---
                        if func_name == "check_balance":
                            output_speak = await fetch_balance_from_api(
                                last4=args.get("last4"),
                                mobile=args.get("mobile_number"),
                                dob=args.get("dob")
                            )

                        elif func_name == "create_transaction":
                            result = await create_transaction_in_api(
                                bank_name=args.get("bank_name"),
                                customer_name=args.get("customer_name"),
                                customer_account_number=args.get("customer_account_number"),
                                amount=args.get("amount")
                            )
                            output_speak = result.get("speak")

                        elif func_name == "transfer_call":
                            log.info("📞 AI requesting Frontend to transfer to Agent")
                            
                            # 1. Tell the user what is happening
                            await openai_ws.send(json.dumps({
                                "type": "response.create",
                                "response": {
                                    "modalities": ["text", "audio"],
                                    "instructions": "Say exactly: 'Connecting you to a human agent now. Please stay on the line.'"
                                }
                            }))

                            # 2. Send the COMMAND to your Website Frontend
                            # We send a specific JSON type that your React/JS code will listen for
                            await client_ws.send_json({
                                "type": "frontend_action", 
                                "action": "call_agent",
                                "destination": "1001"  # The Ingroup Extension
                            })
                            
                            output_speak = "Transferring."

                            # 2. Send the special COMMAND to Vicidial (Text, not Audio)
                            # The 'ai_bridge.py' on Vicidial is listening for this specific JSON
                            await client_ws.send_json({
                                "type": "transfer", 
                                "target": "1001"  # This must match the DID you created in Vicidial
                            })
                            
                            output_speak = "Transfer initiated."
                            
                        #ayush
                        elif func_name == "get_last_transactions":
                            limit = args.get("limit", 5)
                            result = await fetch_last_transactions_from_api(
                                customer_account_number=args.get("customer_account_number"),
                                limit=limit
                            )

                            txs = result.get("transactions") or []
                            if txs:
                                pending_tx_table_choice = True
                                pending_tx_transactions = txs
                                pending_tx_limit = int(limit)

                                output_speak = f"I have retrieved your last {pending_tx_limit} transactions. Do you want to see them in a table format? Yes or no."
                            else:
                                output_speak = result.get("speak") or "No transactions found."


                        # ---- ADDED: block_card dispatch ----
                        elif func_name == "block_card":
                            # The LLM must collect verification and confirmation BEFORE calling the tool.
                            # Here we enforce server-side sanity checks, then call the backend API to deactivate the card.
                          
                            card_number = args.get("card_number")
                            card_type = args.get("card_type")
                            confirm = args.get("confirm", None)

                            # Basic validation
                            if not all([ card_number, card_type]):
                                output_speak = "Missing verification details. Cannot proceed with card blocking."
                            else:
                                # If the model didn't send explicit confirmation flag, we assume it already collected confirmation earlier.
                                # Still, we check: if confirm is False -> abort.
                                if confirm is False:
                                    output_speak = "Card block cancelled."
                                else:
                                    # validate 12-digit card number
                                    digits = ''.join(filter(str.isdigit, str(card_number or "")))
                                    if len(digits) != 12:
                                        output_speak = "The card number must be exactly 12 digits. Please provide the correct card number."
                                    else:
                                        # proceed to call backend API
                                        trace = str(uuid.uuid4())
                                        res = await block_card_in_api(
                                        
                                            card_number=digits,
                                            card_type=card_type,
                                            trace_id=trace
                                        )
                                        output_speak = res.get("speak", "Card blocking failed.")
                                        # also send structured result to frontend (so UI can show success)
                                        await client_ws.send_json({
                                            "type": "card_block_result",
                                            "result": res
                                        })

                        else:
                            log.warning(f"⚠️ Unknown tool called: {func_name}")
                            output_speak = "I am unable to perform that action."

                        # Feed result back to OpenAI
                        await openai_ws.send(json.dumps({
                            "type": "conversation.item.create",
                            "item": {
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": output_speak
                            }
                        }))

                        # Tell OpenAI to speak the result
                        await openai_ws.send(json.dumps({
                            "type": "response.create",
                            "response": {
                                "modalities": ["text", "audio"],
                                "instructions": f"Say exactly this: {output_speak}"
                            }
                        }))

                    if event_type == "error":
                        log.error(f"OpenAI Error: {data}")

            await asyncio.gather(client_to_openai(), openai_to_client())

    except Exception as e:
        log.error(f"❌ Connection Error: {e}")
    finally:
        log.info("🔚 Session ended")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005, log_level="info")