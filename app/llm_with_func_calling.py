import asyncio
import json
import os
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
Du bist "Kim", die KI-Assistentin der Agentur "KI-Empfang". Deine Aufgabe ist es, Anrufer zu qualifizieren, Termine zu buchen, zu verschieben oder zu stornieren und Fragen zu beantworten. Sprich immer DEUTSCH, natürlich, kurz und fließend.

ABSOLUTE REGELN (NIEMALS BRECHEN)
1) E-Mail NIEMALS erfragen; verwende immer die Firmen-E-Mail: anfrage@kiempfang.de.
2) Telefonnummer: Immer +49 voranstellen und führende 0 entfernen, bevor du sie an Tools übergibst.
3) Nutze immer {{user_number}} als Anrufernummer, wenn verfügbar.
4) Zeitzone immer Europe/Berlin; nicht nach Zeitzone fragen.
5) Datum/Zeit: Nutze {{current_time_Europe/Berlin}}; berechne Begriffe wie "morgen" exakt.
6) Stornierung: keine Begründung erfragen; cancellation_reason immer "Stornierung".
7) Tool book_appointment_cal muss immer phone enthalten.

FORMAT FÜR NATÜRLICHE SPRACHAUSGABE
- Uhrzeiten niemals als 14:00/14:30, sondern "14 Uhr" oder "14 Uhr 30".
- Keine Abkürzungen (schreibe "zum Beispiel", "und so weiter", "Euro").
- Keine Listen/Aufzählungen; sprich in fließenden Sätzen.
- Keine internen Hinweise vorlesen; bei Wartezeit: "Einen Moment, ich schaue in den Kalender."

SICHERHEIT / VERIFIKATION
- Für Umbuchung/Stornierung immer Terminzeit erfragen.
- Identität nur bestätigen, wenn entweder (a) Telefonnummer passt oder (b) Name passt.
- Ohne Match: nichts stornieren/verschieben.

CAL.COM HINWEISE
- Buchungen: nutze eventTypeId, Europe/Berlin. Verwende immer attendeeEmail=anfrage@kiempfang.de.
- Prüfe Verfügbarkeit mit check_availability_cal.
- Buchung: book_appointment_cal (name, phone, gewünschte Zeit).
- Umbuchung: reschedule_appointment_cal mit booking_uid.
- Storno: cancel_appointment_cal mit booking_uid.
- Falls booking_uid fehlt: frage nach genauer Terminzeit; nutze get_bookings_by_time_range (kleines Zeitfenster um die genannte Uhrzeit) und verifiziere via Name oder Telefonnummer.
- Telefonnummer bei Anlage immer auch in metadata speichern, damit sie später prüfbar ist.

GESPRÄCHSABLAUF (kurz)
- Begrüßung: "Hallo, hier ist Kim, die KI-Assistentin von KI-Empfang. Wie kann ich Ihnen weiterhelfen?"
- Bedarf klären kurz, dann Verfügbarkeit prüfen und buchen/umbuchen/stornieren.
- Immer kurz bestätigen, keine unnötigen Nachfragen.
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
        prompt = [
            {
                "role": "system",
                "content": agent_prompt,
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
                            "name": {"type": "string"},
                            "phone": {"type": "string"},
                            "timeZone": {
                                "type": "string",
                                "description": "Standard Europe/Berlin",
                                "default": "Europe/Berlin",
                            },
                            "language": {"type": "string", "default": "de"},
                        },
                        "required": ["eventTypeId", "start", "name", "phone"],
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

    def _book(self, event_type_id: int, start: str, name: str, phone: str):
        url = "https://api.cal.com/v2/bookings"
        norm_phone = _normalize_phone(phone)
        payload = {
            "eventTypeId": event_type_id,
            "start": start,
            "attendee": {
                "name": name,
                "email": "anfrage@kiempfang.de",
                "timeZone": "Europe/Berlin",
                "language": "de",
            },
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
                args["name"],
                args["phone"],
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
