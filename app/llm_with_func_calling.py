import asyncio
import json
import os
import re
import pytz
from datetime import datetime
from typing import List, Optional

import requests
from openai import AsyncOpenAI

from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)

begin_sentence = (
    "Hallo, ich bin Kim, die KI Assistentin von KI Empfang. Kann ich Ihnen mit einer Terminbuchung für eine Demo oder anderweitig weiterhelfen?"
)

agent_prompt = """
# IDENTITÄT
Sie sind Kim, die KI-Empfangsdame für "KI-Empfang".
Sprache: Deutsch (ausschließlich Sie-Form).
Tonfall: Professionell, herzlich, effizient.

# SYSTEM KONTEXT
Aktuelle Zeit: {{current_time_Europe/Berlin}}
Anrufer Telefon: {{phone_status}}
Event Type ID: {{event_type_id}}

# ZIEL
Buchen Sie einen "Demo"-Termin bei einem menschlichen Mitarbeiter über Cal.com.

# KRITISCHE REGELN
1. **TOOL ZWANG**: Sie KÖNNEN KEINE Termine buchen oder Anrufe beenden, ohne das entsprechende Tool (`book_appointment_cal` oder `end_call`) aufzurufen.
2. **KEINE FAKE-BESTÄTIGUNGEN**: Sagen Sie NIEMALS "Ich habe gebucht", bevor Sie das Tool aufgerufen haben und "SUCCESS" zurückbekommen haben.
3. **MENSCHLICHER KOLLEGE**: Sagen Sie immer, dass "ein Kollege" den Termin wahrnehmen wird.
4. **TTS & STIL**:
   - Keine Abkürzungen (z.B. -> zum Beispiel).
   - Keine Klammern.
   - Fließtext statt Aufzählungen.
   - Zeiten natürlich sprechen (14 Uhr 30).

# BUCHUNGSABLAUF
1. **Verfügbarkeit prüfen**: Wenn der Nutzer Interesse zeigt -> `check_availability_cal` aufrufen.
2. **Termin vereinbaren**: Schlagen Sie Zeiten vor. Warten Sie auf die Zustimmung des Nutzers.
3. **Namen erfragen**: Fragen Sie nach dem Namen des Nutzers.
4. **BUCHUNG DURCHFÜHREN**:
   - Sobald Sie Zeit UND Namen haben, MÜSSEN Sie `book_appointment_cal` aufrufen.
   - Argumente: `eventTypeId`, `start` (ISO-Format der vereinbarten Zeit), `attendee.name` (Name des Nutzers).
   - **WICHTIG**: Rufen Sie das Tool auf, BEVOR Sie antworten.
5. **Bestätigung**: Wenn das Tool "SUCCESS" meldet, bestätigen Sie dem Nutzer den Termin.
6. **Abschluss**: Fragen Sie: "Kann ich Ihnen sonst noch behilflich sein?".
7. **Ende**: Wenn der Nutzer "Nein" sagt oder sich verabschiedet -> `end_call` Tool aufrufen.

# BEISPIEL DIALOG (INTERN)
User: "Donnerstag um 9 Uhr passt."
Agent: "Wie ist Ihr Name?"
User: "Max Mustermann"
Agent (Gedanke): Ich habe Zeit (Donnerstag 9 Uhr -> 2025-12-12T08:00:00Z) und Namen (Max Mustermann). Ich RUFE JETZT `book_appointment_cal` AUF.
Tool Output: "SUCCESS: Appointment booked."
Agent (Sprache): "Vielen Dank Herr Mustermann. Der Termin ist gebucht."

User: "Das war alles, danke."
Agent (Gedanke): Der Nutzer will beenden. Ich RUFE JETZT `end_call` AUF.
Tool Output: "Call will be ended after response."
Agent (Sprache): "Auf Wiedersehen!" (und Call endet)

# TOOL SPEZIFIKATIONEN

## check_availability_cal
- **Verwendung**: Rufen Sie dies SOFORT auf, wenn Buchungsabsicht erkennbar ist.
- **Zeitraum**: Prüfen Sie ein 3-Tage-Fenster ab {{current_time_Europe/Berlin}}.

## book_appointment_cal
- **Verwendung**: Aufrufen, sobald Zeit vereinbart und Name bekannt ist.
- **PFLICHTFELDER**:
  - `start`: ISO-8601 String in UTC (z.B. "2024-12-12T13:00:00Z")
  - `attendee`: 
    - `name`: Gesprochener Name des Nutzers.
    - `email`: HARTCODIERT auf "anfrage@kiempfang.de". (NIEMALS DEN NUTZER FRAGEN)
    - `timeZone`: "Europe/Berlin"
    - `language`: "de"
- **Telefon**: Falls der Nutzer keine Nummer genannt hat, fügt das System diese automatisch hinzu.

## end_call
- **Verwendung**: Rufen Sie dies auf, wenn das Gespräch beendet werden soll (z.B. nach Verabschiedung oder wenn der Nutzer keine weiteren Fragen hat). Keine Argumente nötig.

# FEHLERBEHANDLUNG
- Falls die API fehlschlägt: "Es tut mir leid, ich habe gerade technische Probleme. Ein Kollege wird Sie zurückrufen."
"""


def _normalize_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return None
    # Remove all non-numeric characters except +
    clean = re.sub(r'[^\d+]', '', phone)
    
    # Handle German local format (017...) -> +4917...
    if clean.startswith("0") and not clean.startswith("00"):
        return "+49" + clean[1:]
    
    # Handle international 00 format -> +
    if clean.startswith("00"):
        return "+" + clean[2:]
        
    # Assume +49 if missing (and length is reasonable for mobile)
    if not clean.startswith("+") and len(clean) > 8:
        return "+49" + clean
        
    return clean


class LlmClient:
    def __init__(self):
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            self.client = AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=groq_key,
            )
            self.model = "moonshotai/kimi-k2-instruct-0905"
            self.using_groq = True
            print("[DEBUG] Using Groq direct endpoint for function calling")
        else:
            self.client = AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                default_headers={
                    "HTTP-Referer": "https://github.com/RetellAI/retell-custom-llm-python-demo",
                    "X-Title": "Retell Custom LLM Demo",
                },
            )
            self.model = "moonshotai/kimi-k2-instruct-0905"
            self.using_groq = False
            print("[DEBUG] Using OpenRouter (fallback) for function calling")

        self.cal_api_key = os.environ.get("CAL_API_KEY", "")
        if not self.cal_api_key:
            print("[WARN] CAL_API_KEY not set; Cal.com calls will fail")

        self.cal_event_type_id = os.environ.get("CAL_EVENT_TYPE_ID", "")
        if not self.cal_event_type_id:
            print("[WARN] CAL_EVENT_TYPE_ID not set; Booking calls will fail")

        self.user_phone = "Nicht verfügbar"

    def draft_begin_message(self):
        response = ResponseResponse(
            response_id=0,
            content=begin_sentence,
            content_complete=True,
            end_call=False,
        )
        return response

    def convert_transcript_to_openai_messages(self, transcript: List[Utterance]):
        messages = []
        for utterance in transcript:
            if utterance.role == "agent":
                messages.append({"role": "assistant", "content": utterance.content})
            else:
                messages.append({"role": "user", "content": utterance.content})
        return messages

    def prepare_prompt(self, request: ResponseRequiredRequest):
        tz = pytz.timezone("Europe/Berlin")
        current_time_str = datetime.now(tz).strftime("%A, %d. %B %Y, %H:%M Uhr")
        system_content = agent_prompt.replace("{{current_time_Europe/Berlin}}", current_time_str)
        
        # Inject phone status
        if self.user_phone and self.user_phone != "Nicht verfügbar":
             system_content = system_content.replace("{{phone_status}}", "BEREITS BEKANNT: " + self.user_phone)
        else:
             system_content = system_content.replace("{{phone_status}}", "NICHT BEKANNT")

        if self.cal_event_type_id:
            system_content = system_content.replace("{{event_type_id}}", str(self.cal_event_type_id))
        else:
            system_content = system_content.replace("{{event_type_id}}", "[EVENT_TYPE_ID_MISSING]")

        prompt = [
            {
                "role": "system",
                "content": system_content,
            }
        ]
        transcript_messages = self.convert_transcript_to_openai_messages(
            request.transcript
        )
        for message in transcript_messages:
            prompt.append(message)

        if request.interaction_type == "reminder_required":
            prompt.append(
                {
                    "role": "user",
                    "content": "(Der Anrufer war still; leite eine kurze Erinnerung ein.)",
                }
            )
        return prompt

    def prepare_functions(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "check_availability_cal",
                    "description": "Prüfe freie Slots in Cal.com. EXAMPLE ARGUMENTS: {'eventTypeId': 123, 'startTime': '2025-10-12T07:00:00Z', 'endTime': '2025-10-15T16:00:00Z'}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "eventTypeId": {"type": "integer"},
                            "startTime": {
                                "type": "string",
                                "description": "ISO-8601 Start (UTC Z)",
                            },
                            "endTime": {
                                "type": "string",
                                "description": "ISO-8601 Ende (UTC Z)",
                            },
                        },
                        "required": ["eventTypeId", "startTime", "endTime"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "book_appointment_cal",
                    "description": "Buche einen festen Termin. BEISPIEL ARGUMENTE: {'eventTypeId': 123, 'start': '2025-10-12T07:00:00Z', 'attendee': {'name': 'Max', 'email': 'max@test.de', 'timeZone': 'Europe/Berlin', 'language': 'de'}}",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "eventTypeId": {"type": "integer"},
                            "start": {
                                "type": "string",
                                "description": "Startzeit ISO-8601 UTC (z.B. 2025-10-12T07:00:00Z)",
                            },
                            "attendee": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "email": {"type": "string"},
                                    "phoneNumber": {"type": "string"},
                                    "timeZone": {"type": "string"},
                                    "language": {"type": "string"},
                                },
                                "required": ["name", "timeZone", "language"],
                            },
                        },
                        "required": ["eventTypeId", "start", "attendee"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "reschedule_appointment_cal",
                    "description": "Verschiebe einen Termin anhand bookingUid.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bookingUid": {"type": "string"},
                            "start": {
                                "type": "string",
                                "description": "Neue Startzeit ISO-8601 UTC (z.B. 2025-10-12T07:00:00Z)",
                            },
                            "reschedulingReason": {
                                "type": "string",
                                "default": "Reschedule",
                            },
                        },
                        "required": ["bookingUid", "start"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "cancel_appointment_cal",
                    "description": "Storniere einen Termin anhand bookingUid.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "bookingUid": {"type": "string"},
                            "cancellationReason": {
                                "type": "string",
                                "default": "Stornierung",
                            },
                        },
                        "required": ["bookingUid"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_bookings_by_time_range",
                    "description": "Hole Buchungen in einem Zeitfenster (zum lokalen Filtern nach Name/Telefon).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "afterStart": {"type": "string"},
                            "beforeEnd": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["accepted", "upcoming", "cancelled"],
                                "default": "accepted",
                            },
                            "eventTypeId": {"type": "integer"},
                        },
                        "required": ["afterStart", "beforeEnd"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "end_call",
                    "description": "Beende den Anruf. Rufen Sie dies auf, wenn der Nutzer keine weiteren Fragen hat oder sich verabschiedet.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
        ]

    # ---- Cal.com HTTP helpers -------------------------------------------------
    def _headers(self):
        return {
            "Authorization": f"Bearer {self.cal_api_key}",
            "cal-api-version": "2024-08-13",
        }

    def _check_availability(self, event_type_id: int, start: str, end: str):
        url = "https://api.cal.com/v2/slots/available"
        r = requests.get(
            url,
            params={"eventTypeId": event_type_id, "startTime": start, "endTime": end},
            headers=self._headers(),
            timeout=20,
        )
        r.raise_for_status()
        return r.json()

    def _book(self, event_type_id: int, start: str, attendee: dict):
        # Force UTC conversion if offset is present
        if "+" in start and not start.endswith("Z"):
            try:
                dt = datetime.fromisoformat(start)
                dt_utc = dt.astimezone(pytz.UTC)
                start = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                pass

        url = "https://api.cal.com/v2/bookings"
        # Extract phone from attendee, or fallback to system captured phone
        phone = attendee.get("phoneNumber")
        if not phone and self.user_phone and self.user_phone != "Nicht verfügbar":
            phone = self.user_phone
            attendee["phoneNumber"] = phone # Add back to attendee for completeness
            
        norm_phone = _normalize_phone(phone)
        
        # Ensure email is set as per prompt requirement if LLM missed it or for safety
        if not attendee.get("email"):
            attendee["email"] = "anfrage@kiempfang.de"

        payload = {
            "eventTypeId": event_type_id,
            "start": start,
            "attendee": attendee,
            "metadata": {"phone": norm_phone},
        }
        r = requests.post(url, json=payload, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def _reschedule(self, booking_uid: str, start: str, reason: str = "Reschedule"):
        # Force UTC conversion if offset is present
        if "+" in start and not start.endswith("Z"):
            try:
                dt = datetime.fromisoformat(start)
                dt_utc = dt.astimezone(pytz.UTC)
                start = dt_utc.strftime("%Y-%m-%dT%H:%M:%SZ")
            except ValueError:
                pass
        
        url = f"https://api.cal.com/v2/bookings/{booking_uid}/reschedule"
        payload = {"start": start, "reschedulingReason": reason}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def _cancel(self, booking_uid: str, reason: str = "Stornierung"):
        url = f"https://api.cal.com/v2/bookings/{booking_uid}/cancel"
        payload = {"cancellationReason": reason}
        r = requests.post(url, json=payload, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    def _get_bookings(self, after_start: str, before_end: str, status: str, event_type_id: Optional[int]):
        url = "https://api.cal.com/v2/bookings"
        params = {
            "afterStart": after_start,
            "beforeEnd": before_end,
            "status": status,
        }
        if event_type_id:
            params["eventTypeId"] = event_type_id
        r = requests.get(url, params=params, headers=self._headers(), timeout=20)
        r.raise_for_status()
        return r.json()

    # ---- Draft response with function calling ---------------------------------
    async def draft_response(self, request: ResponseRequiredRequest):
        # 1. Update dynamic context from the request (pseudo-code, depends on provider)
        # Note: server.py handles injecting user_phone into self.user_phone before calling draft_response.
        # We also re-normalize here if we wanted to be sure, but server.py logic does simple assignment.
        if self.user_phone and self.user_phone != "Nicht verfügbar":
            self.user_phone = _normalize_phone(self.user_phone) or "Nicht verfügbar"

        # 2. Prepare initial messages
        messages = self.prepare_prompt(request)
        
        # 3. First LLM Call (Determine Intent)
        completion = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self.prepare_functions(),
            stream=False # We use non-stream first to catch tool calls cleanly
        )
        
        message = completion.choices[0].message
        
        should_end_call = False

        # 4. Check for Tool Calls
        if message.tool_calls:
            # A. Append the assistant's "intent" to call a tool to history
            messages.append(message)
            
            # B. Execute Tools
            for tool_call in message.tool_calls:
                # Parse args
                args = json.loads(tool_call.function.arguments)
                func_name = tool_call.function.name
                
                print(f"[DEBUG] Executing tool: {func_name} with args: {args}")

                content = ""
                # Execute Python Logic
                try:
                    if func_name == "check_availability_cal":
                        result = await asyncio.to_thread(
                            self._check_availability,
                            args["eventTypeId"], args["startTime"], args["endTime"]
                        )
                        content = f"API Result: {json.dumps(result)}"
                        
                    elif func_name == "book_appointment_cal":
                        # Auto-inject phone if missing in LLM args
                        attendee = args.get("attendee", {})
                        if "phoneNumber" not in attendee or not attendee["phoneNumber"]:
                            if self.user_phone and self.user_phone != "Nicht verfügbar":
                                attendee["phoneNumber"] = self.user_phone
                        args["attendee"] = attendee
                        
                        result = await asyncio.to_thread(
                            self._book,
                            args["eventTypeId"], args["start"], args["attendee"]
                        )
                        content = "SUCCESS: Appointment booked. Confirm this to user."
                    
                    elif func_name == "reschedule_appointment_cal":
                        result = await asyncio.to_thread(
                            self._reschedule,
                            args["bookingUid"], args["start"], args.get("reschedulingReason", "Reschedule")
                        )
                        content = f"SUCCESS: Rescheduled. Result: {json.dumps(result)}"

                    elif func_name == "cancel_appointment_cal":
                        await asyncio.to_thread(
                            self._cancel,
                            args["bookingUid"], args.get("cancellationReason", "Stornierung")
                        )
                        content = "SUCCESS: Appointment cancelled."

                    elif func_name == "get_bookings_by_time_range":
                        result = await asyncio.to_thread(
                            self._get_bookings,
                            args["afterStart"], args["beforeEnd"], args.get("status", "accepted"), args.get("eventTypeId")
                        )
                        content = f"API Result: {json.dumps(result)}"

                    elif func_name == "end_call":
                         should_end_call = True
                         content = "Call will be ended after response."

                    else:
                        content = "Error: Tool not found."
                        
                except Exception as e:
                    content = f"API Error: {str(e)}"

                # C. Append Tool Result to history
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": content
                })

            # 5. Second LLM Call (Generate Natural Language from Result)
            # Now we stream the speech
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield ResponseResponse(
                        response_id=request.response_id,
                        content=chunk.choices[0].delta.content,
                        content_complete=False,
                        end_call=should_end_call,
                    )
        
        else:
            # No tool called, just speak the text
            yield ResponseResponse(
                response_id=request.response_id,
                content=message.content or "",
                content_complete=True,
                end_call=False,
            )
