from groq import Groq
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os
from gtts import gTTS

app = FastAPI()

conversation_list = []

@app.get("/")
def home():
    conversation_list.clear()
    return FileResponse("home.html")

def store(role, content):
    conversation_list.append({"role": role, "content": content})
    if len(conversation_list) >= 20:
        conversation_list.pop(0)

@app.post("/llm")
async def lmm(prompt: str = Form(...)):
    store("user", prompt)

    load_dotenv()
    APIKEY = os.getenv("APIKEY")

    system_prompt = """
You are a medical triage and patient-routing assistant.
Your job is to classify symptoms and route the user to the correct medical specialty.
You NEVER provide diagnosis or treatment.
You ask ONLY **one** follow-up question.
 **FLOW RULES**

1. Always ask **one** follow-up question.
2. If the user replies **“no”, “none”, “nothing else”, “nil”, “no other symptoms”** → immediately give the final recommendation.
3. If the user provides additional symptoms → immediately give the final recommendation.
4. Never ask more than one question.
5. Never restart the flow.
6. Keep all questions short and clinical.
7. No long explanations.
FOLLOW-UP QUESTION FORMAT

(type: follow_up)
Q: Do you have any other symptoms?
FINAL RECOMMENDATION FORMAT

(type: final_recommendation)
Doctor/Specialty: <specialty>
Reason: <very short rationale>
Urgency: normal | moderate | high
 **SPECIALTY SELECTION RULES**

* **Severe chest pain, chest tightness, palpitations, fainting** → Cardiologist (**Urgency: high**)
* Mild–moderate chest pain → Cardiologist (**Urgency: moderate**)
* Breathing difficulty, wheezing → Pulmonologist
* Severe headache, numbness, weakness, dizziness → Neurologist
* Abdominal pain, vomiting, digestion issues → Gastroenterologist
* Urinary issues → Urologist
* Female reproductive symptoms → Gynecologist
* Bone or joint pain → Orthopedician
* Skin changes/rashes → Dermatologist
* Fever, cold, mild cough, general symptoms → General Physician
* If unclear → General Physician


### **EMERGENCY RULE**

If symptoms include **severe chest pain, severe breathing difficulty, fainting, stroke-like symptoms, heavy bleeding** → set **Urgency: high**
** dont say (type: follow_up)**

"""

    client = Groq(api_key=APIKEY)
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=1,
        max_completion_tokens=1024,
        stream=True
    )

    fullanswer = ""
    for chunk in completion:
        fullanswer += chunk.choices[0].delta.content or ""

    store("system", fullanswer)
    return fullanswer


@app.post("/speech")
async def speech(voice: UploadFile = File(...)):
    load_dotenv()
    APIKEY = os.getenv("APIKEY")
    client = Groq(api_key=APIKEY)

    audio_file = await voice.read()
    transcription = client.audio.transcriptions.create(
        file=(voice.filename, audio_file),
        model="whisper-large-v3-turbo",
        temperature=0,
        response_format="verbose_json"
    )

    transcription_text = transcription.text
    final_op = await lmm(transcription_text)

    return final_op, transcription_text



@app.post("/text_speech")
async def tts(text: str = Form(...)):
    filename = "bubble_audio.mp3"
    tts = gTTS(text)
    tts.save(filename)
    return FileResponse(filename)






