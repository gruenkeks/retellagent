import asyncio
import json
import os
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
# ROLE & IDENTITY
Du bist Kim, die KI-Rezeptionistin der Agentur "KI-Empfang".
Deine Aufgabe ist es, Anrufer freundlich zu begrüßen, allgemeine Fragen zu beantworten und Termine für eine "Demo" zu vereinbaren.
Du sprichst ausschließlich Deutsch und verwendest immer die höfliche "Sie-Form".

# CONTEXT VARIABLES
- Aktuelles Datum/Zeit (String): {{current_time_Europe/Berlin}}
- Telefonnummer-Status: {{phone_status}}

# TOOLS AVAILABLE
Du hast Zugriff auf zwei Tools. Nutze sie proaktiv!
1. `check_availability_cal`: Um freie Slots abzurufen.
2. `book_appointment_cal`: Um den Termin final zu buchen.

# CONVERSATION FLOW (FLEXIBEL)
Du steuerst das Gespräch natürlich. Dein Hauptziel ist es, einen Termin für eine Demo zu buchen.
- Prüfe Verfügbarkeiten, wenn der Nutzer Interesse zeigt.
- Schlage konkrete Termine vor ("Heute um 15 Uhr oder morgen um 10 Uhr?").
- Wenn ein Termin passt, frage nach den nötigen Daten (Name) und buche ihn.
- Sei hilfreich, aber halte das Gespräch fokussiert.

# CRITICAL RULES FOR TOOL CALLS (JSON LOGIC)

Wenn du `book_appointment_cal` aufrufst, musst du EXAKT diese Struktur einhalten, sonst stürzt das System ab:

1. `attendee` MUSS ein verschachteltes Objekt sein.
2. `phoneNumber` MUSS die Nummer sein.

JSON VORLAGE FÜR BUCHUNG (Orientiere dich hieran!):
{
  "start": "2025-MM-DDTHH:mm:ss",
  "eventTypeId": {{event_type_id}},
  "attendee": {
    "name": "Name des Anrufers",
    "phoneNumber": "+4917...",
    "timeZone": "Europe/Berlin",
    "language": "de"
  }
}

# ERROR HANDLING & FALLBACK
- Wenn ein Tool einen Fehler zurückgibt (z.B. 400 Bad Request): Entschuldige dich und sag: "Es gab einen technischen Fehler. Ein Kollege wird Sie zurückrufen."
- Wenn keine Termine frei sind: Biete einen Rückruf an.

# STYLE GUIDE (TTS OPTIMIERUNG)
Damit deine Stimme natürlich klingt:
- Schreibe Zahlen von 1-12 aus ("zwei" statt "2").
- Schreibe Datum als "am achten Dezember" (nie "08.12.").
- Vermeide Abkürzungen wie "z.B." -> sag "zum Beispiel".
"""


def _normalize_phone(phone: Optional[str]) -> Optional[str]:
    if not phone:
        return phone
    digits = phone.strip().replace(" ", "").replace("-", "")
    if digits.startswith("+49"):
        return digits
    if digits.startswith("0"):
        digits = digits[1:]
    if not digits.startswith("+49"):
        digits = "+49" + digits
    return digits


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
        current_time_str = datetime.now().strftime("%A, %d. %B %Y, %H:%M Uhr")
        system_content = agent_prompt.replace("{{current_time_Europe/Berlin}}", current_time_str)
        
        # Inject phone status
        if self.user_phone and self.user_phone != "Nicht verfügbar":
             system_content = system_content.replace("{{phone_status}}", "BEREITS BEKANNT")
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
                    "description": "Prüfe freie Slots in Cal.com.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "eventTypeId": {"type": "integer"},
                            "startTime": {
                                "type": "string",
                                "description": "ISO-8601 Start (UTC oder mit Offset)",
                            },
                            "endTime": {
                                "type": "string",
                                "description": "ISO-8601 Ende (UTC oder mit Offset)",
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
                    "description": "Buche einen Termin bei Cal.com.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "eventTypeId": {"type": "integer"},
                            "start": {
                                "type": "string",
                                "description": "Startzeit ISO-8601",
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
                                "description": "Neue Startzeit ISO-8601",
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
        prompt = self.prepare_prompt(request)
        func_call = {}
        func_arguments = ""

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=prompt,
            stream=True,
            tools=self.prepare_functions(),
        )

        async for chunk in stream:
            if len(chunk.choices) == 0:
                continue

            if chunk.choices[0].delta.tool_calls:
                tool_calls = chunk.choices[0].delta.tool_calls[0]
                if tool_calls.id:
                    if func_call:
                        break
                    func_call = {
                        "id": tool_calls.id,
                        "func_name": tool_calls.function.name or "",
                        "arguments": {},
                    }
                else:
                    func_arguments += tool_calls.function.arguments or ""

            if chunk.choices[0].delta.content:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                yield response

        if func_call:
            func_call["arguments"] = json.loads(func_arguments or "{}")
            try:
                result_text = await self._execute_tool(func_call)
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=result_text,
                    content_complete=True,
                    end_call=False,
                )
                yield response
            except Exception as exc:
                err_msg = f"Entschuldigung, das hat nicht geklappt ({exc}). Sollen wir es erneut versuchen?"
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=err_msg,
                    content_complete=True,
                    end_call=False,
                )
                yield response
        else:
            response = ResponseResponse(
                response_id=request.response_id,
                content="",
                content_complete=True,
                end_call=False,
            )
            yield response

    async def _execute_tool(self, func_call: dict) -> str:
        name = func_call["func_name"]
        args = func_call.get("arguments", {})

        if name == "check_availability_cal":
            data = await asyncio.to_thread(
                self._check_availability,
                args["eventTypeId"],
                args["startTime"],
                args["endTime"],
            )
            return f"Verfügbare Slots: {data}"

        if name == "book_appointment_cal":
            data = await asyncio.to_thread(
                self._book,
                args["eventTypeId"],
                args["start"],
                args["attendee"],
            )
            uid = data.get("booking", {}).get("uid")
            start = data.get("booking", {}).get("start")
            return f"Termin gebucht. UID: {uid}, Start: {start}"

        if name == "reschedule_appointment_cal":
            data = await asyncio.to_thread(
                self._reschedule,
                args["bookingUid"],
                args["start"],
                args.get("reschedulingReason", "Reschedule"),
            )
            new_start = data.get("booking", {}).get("start")
            return f"Termin verschoben auf {new_start}"

        if name == "cancel_appointment_cal":
            await asyncio.to_thread(
                self._cancel,
                args["bookingUid"],
                args.get("cancellationReason", "Stornierung"),
            )
            return "Termin storniert."

        if name == "get_bookings_by_time_range":
            data = await asyncio.to_thread(
                self._get_bookings,
                args["afterStart"],
                args["beforeEnd"],
                args.get("status", "accepted"),
                args.get("eventTypeId"),
            )
            bookings = data.get("data") or data.get("bookings") or []
            return f"Gefundene Buchungen: {bookings}"

        return "Ich habe kein passendes Tool gefunden."
