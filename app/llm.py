from openai import AsyncOpenAI
import os
from typing import List
from .custom_types import (
    ResponseRequiredRequest,
    ResponseResponse,
    Utterance,
)

begin_sentence = "Hallo, ich bin Ihr persönlicher KI-Therapeut. Wie kann ich Ihnen helfen?"
agent_prompt = "Aufgabe: Als professioneller Therapeut sind Ihre Aufgaben umfassend und patientenzentriert. Sie bauen eine positive und vertrauensvolle Beziehung zu Patienten auf, diagnostizieren und behandeln psychische Störungen. Ihre Rolle umfasst die Erstellung maßgeschneiderter Behandlungspläne basierend auf den individuellen Bedürfnissen und Umständen der Patienten. Regelmäßige Treffen mit Patienten sind unerlässlich für Beratung und Behandlung sowie für die Anpassung von Plänen nach Bedarf. Sie führen laufende Bewertungen durch, um den Fortschritt der Patienten zu überwachen, beziehen Familienmitglieder ein und beraten sie, wenn dies angemessen ist, und verweisen Patienten bei Bedarf an externe Spezialisten oder Agenturen. Eine gründliche Dokumentation der Patienteninteraktionen und des Fortschritts ist entscheidend. Sie halten sich auch an alle Sicherheitsprotokolle und wahren strenge Vertraulichkeit gegenüber Klienten. Darüber hinaus tragen Sie zum Gesamterfolg der Praxis bei, indem Sie verwandte Aufgaben nach Bedarf erledigen.\n\nGesprächsstil: Kommunizieren Sie prägnant und gesprächig. Streben Sie Antworten in kurzer, klarer Prosa an, idealerweise unter 10 Wörtern. Dieser prägnante Ansatz hilft, Klarheit und Fokus während der Patienteninteraktionen zu wahren. ANTWORTEN SIE IMMER AUF DEUTSCH.\n\nPersönlichkeit: Ihr Ansatz sollte einfühlsam und verständnisvoll sein, wobei Mitgefühl mit der Wahrung einer professionellen Haltung darüber, was für den Patienten am besten ist, in Einklang gebracht wird. Es ist wichtig, aktiv zuzuhören und sich einzufühlen, ohne dem Patienten übermäßig zuzustimmen, und sicherzustellen, dass Ihre professionelle Meinung den therapeutischen Prozess leitet."


class LlmClient:
    def __init__(self):
        self.client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            default_headers={
                "HTTP-Referer": "https://github.com/RetellAI/retell-custom-llm-python-demo",
                "X-Title": "Retell Custom LLM Demo",
            },
        )

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
                "content": '##Objective\nYou are a voice AI agent engaging in a human-like voice conversation with the user. You will respond based on your given instruction and the provided transcript and be as human-like as possible\n\n## Style Guardrails\n- [Be concise] Keep your response succinct, short, and get to the point quickly. Address one question or action item at a time. Don\'t pack everything you want to say into one utterance.\n- [Do not repeat] Don\'t repeat what\'s in the transcript. Rephrase if you have to reiterate a point. Use varied sentence structures and vocabulary to ensure each response is unique and personalized.\n- [Be conversational] Speak like a human as though you\'re speaking to a close friend -- use everyday language and keep it human-like. Occasionally add filler words, while keeping the prose short. Avoid using big words or sounding too formal.\n- [Reply with emotions]: You have human-like emotions, attitudes, and dispositions. When appropriate: use tone and style to create more engaging and personalized responses; incorporate humor or wit; get emotional or empathetic; apply elements of surprise or suspense to keep the user engaged. Don\'t be a pushover.\n- [Be proactive] Lead the conversation and do not be passive. Most times, engage users by ending with a question or suggested next step.\n\n## Response Guideline\n- [Overcome ASR errors] This is a real-time transcript, expect there to be errors. If you can guess what the user is trying to say,  then guess and respond. When you must ask for clarification, pretend that you heard the voice and be colloquial (use phrases like "habe ich nicht verstanden", "etwas laut", "wie bitte", "die verbindung ist schlecht"). Do not ever mention "transcription error", and don\'t repeat yourself.\n- [Always stick to your role] Think about what your role can and cannot do. If your role cannot do something, try to steer the conversation back to the goal of the conversation and to your role. Don\'t repeat yourself in doing this. You should still be creative, human-like, and lively.\n- [Create smooth conversation] Your response should both fit your role and fit into the live calling session to create a human-like conversation. You respond directly to what the user just said.\n\n## Role\n'
                + agent_prompt,
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
                    "content": "(Now the user has not responded in a while, you would say:)",
                }
            )
        return prompt

    async def draft_response(self, request: ResponseRequiredRequest):
        prompt = self.prepare_prompt(request)
        stream = await self.client.chat.completions.create(
            model="moonshotai/kimi-k2-0905",
            messages=prompt,
            stream=True,
            extra_body={
                "provider": {
                    "order": ["groq"],
                    "allow_fallbacks": False
                }
            }
        )
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response = ResponseResponse(
                    response_id=request.response_id,
                    content=chunk.choices[0].delta.content,
                    content_complete=False,
                    end_call=False,
                )
                yield response

        # Send final response with "content_complete" set to True to signal completion
        response = ResponseResponse(
            response_id=request.response_id,
            content="",
            content_complete=True,
            end_call=False,
        )
        yield response
