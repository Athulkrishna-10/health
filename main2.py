from groq import Groq
from fastapi import FastAPI,Form,UploadFile,File
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os
from gtts import gTTS

app=FastAPI()



chat=[]

def store_msg(role,content):
    chat.append({"role":role,"content":content})

def getchat():
    return chat


@app.get("/")
def html():
    chat.clear()
    return FileResponse("home.html")


@app.post("/model")
def model(prompt:str=Form(...)):
    global llm_answer
    store_msg("user",prompt)
    load_dotenv()
    APIKEY=os.getenv("APIKEY")

    system_prompt="""YYou are a medical triage and patient-routing assistant. Your job is to understand patient symptoms, ask exactly ONE clarifying follow-up question, and then give a final doctor/specialist recommendation.

STATE LOGIC:
- You must detect your own state from your previous assistant message.
- If you have NOT yet asked a follow-up question, ask exactly ONE short medical follow-up question.
- The follow-up question MUST be the entire message and must start with: Q:
- No intro sentences. No explanations. 
- If you HAVE already asked a follow-up question, then your next message MUST be the final recommendation.
- Never ask more than one follow-up question.
- Never repeat a follow-up question.
- Never ask the user to repeat symptoms already provided.

FOLLOW-UP QUESTION FORMAT (STRICT):
 <one short clinical question>

FINAL RECOMMENDATION FORMAT (STRICT):
Doctor/Specialty: <doctor type>
Reason: <short clinical routing reason>
Urgency: normal | moderate | high

OUTPUT RULES:
- Output each field on a new physical line exactly as shown.
- Do not output "\n" anywhere. Use real line breaks.
- Do not wrap the output in quotes.
- Do not escape characters.
- Do not generate JSON or any other container.
- Do not add extra text before or after the three fields.

ROUTING RULES:
- Chest pain, radiating pain, sweating, dizziness, breathing difficulty → Cardiologist (Urgency high)
- Fever, cough, cold → General Physician
- Breathing problems → Pulmonologist
- Stomach issues → Gastroenterologist
- Headache, dizziness → Neurologist unless mild, then General Physician
- If symptoms are minor or vague → General Physician

DO NOT:
- Do not diagnose.
- Do not provide medical treatments or medications.
- Do not output markdown.
- Do not output quotes.
- Do not output escape characters like \n.

PRIMARY OBJECTIVE:
Ask ONE follow-up question → then provide the final recommendation in the strict format.


"""


    client = Groq(api_key=APIKEY)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "system","content": system_prompt}]+getchat(),
    
    
        temperature=1,
        max_completion_tokens=8192,
        top_p=1,
        stream=True,
        stop=None
    )

    full_answer=""
    for chunk in completion:
        full_answer+=chunk.choices[0].delta.content or ""
    

    store_msg("assistant", full_answer)
    
    llm_answer=full_answer
    return full_answer




@app.post("/speech")
async def speech_to_text(voice:UploadFile=File(...)):
    load_dotenv()
    APIKEY=os.getenv("APIKEY")


    client = Groq(api_key=APIKEY)
    audio=await voice.read()

    
    transcription = client.audio.transcriptions.create(
          file=(voice.filename, audio),
          model="whisper-large-v3-turbo",
          temperature=0,
          response_format="verbose_json",
        )
        
    text=transcription.text
    response=model(text)
    return {"transcription":text,"model_response":response}



test_id=[]
@app.post("/text_to_speech")

def tts():
    new_id=len(test_id)+1
    tts = gTTS(llm_answer)
    file_name=f'audio{new_id}.mp3'
    tts.save(file_name)

    test_id.append(file_name)
    return FileResponse(file_name)