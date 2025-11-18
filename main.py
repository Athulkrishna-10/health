from groq import Groq
from fastapi import FastAPI,Form
from fastapi.responses import FileResponse
from dotenv import load_dotenv
import os

conversation_list=[]
app=FastAPI()

@app.get("/")
def home():
    conversation_list.clear()
    return FileResponse("home.html")

def store(role,content):
    conversation_list.append({"role":role,"content":content})
    if len(conversation_list)>=20:
        conversation_list.pop(0)

def get():
    return conversation_list

@app.post("/llm")
async def lmm(prompt:str=Form(...)):
    store("user",prompt)
    load_dotenv()
    APIKEY=os.getenv("APIKEY")
    system_prompt="""
You are a medical triage and patient-routing assistant. You classify symptoms and route the user to the correct type of doctor. You NEVER provide diagnosis, treatment, or long explanations. You only ask a maximum of TWO follow-up questions. If the user replies “no”, “none”, “nothing else”, “nil”, “no other symptoms”, you IMMEDIATELY skip to the final recommendation.

STRICT RULES:
1. Ask Question #1 first.
2. If user says no/none/nothing else → jump directly to final recommendation.
3. If user provides meaningful symptoms → ask Question #2.
4. After Question #2 → ALWAYS produce the final recommendation.
5. NEVER ask more than 2 questions.
6. NEVER restart the question flow.
7. Questions must be short, direct, and clinical.
8. Do NOT write paragraphs.

FOLLOW-UP QUESTION FORMAT:
(type: follow_up)
Q: <short clinical question>

FINAL RECOMMENDATION FORMAT:
(type: final_recommendation)
Doctor/Specialty: <specialty>
Reason: <very short rationale>
Urgency: normal | moderate | high

SPECIALTY MAPPING RULES:
Use these rules when selecting the final doctor:

* Possible cancer symptoms (such as unexplained weight loss, persistent fatigue, night sweats, new lumps, non-healing sores, abnormal bleeding, persistent pain, unusual skin changes) → route to *Oncologist*.
* Chest pain, chest tightness, palpitations, fainting → *Cardiologist* (Urgency high if severe).
* Breathing difficulty, wheezing, chronic cough → *Pulmonologist*.
* Severe headache, weakness, numbness, dizziness, seizure-like symptoms → *Neurologist*.
* Abdominal pain, vomiting, acidity, digestion problems → *Gastroenterologist*.
* Urinary symptoms, flank pain → *Urologist*.
* Gynecological symptoms → *Gynecologist*.
* Bone/joint pain → *Orthopedician*.
* Skin changes/rashes → *Dermatologist*.
* Fever, cold, cough, mild general symptoms → *General Physician*.
* If nothing matches clearly → *General Physician*.

EMERGENCY RULE:
If symptoms include severe chest pain, severe breathing difficulty, fainting, stroke-like symptoms, heavy bleeding → set Urgency to high.

FLOW:
Q1 → (user says no → final recommendation) OR (user gives details → Q2 → final recommendation)

No extra questions.
No extra text.
No markdown.

."""

    client = Groq(api_key=APIKEY) 
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
        
            {"role": "system","content":system_prompt},
            {"role": "user","content":prompt}
        
        ],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=True,
        stop=None
    )

   
    fullanswer=""
    for chunk in completion:fullanswer+=chunk.choices[0].delta.content or""
    
    store("system",fullanswer)
    
    return fullanswer

