import cv2
from ultralytics import YOLO
import easyocr
import os
import time
from dotenv import load_dotenv
from groq import Groq
import json
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
import pickle
import re
import warnings

# Ignore unnecessary warnings in the terminal
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# CONFIG: MODEL NAME (change here only, one place)
# ==========================================
GROQ_MODEL = "openai/gpt-oss-120b"   # llama-3.3-70b-versatile deprecated by Groq (17 June 2026)

# ==========================================
# PHASE 1: GLOBAL INITIALIZATIONS
# ==========================================
print("Loading AI Models (YOLO + EasyOCR)... This might take a moment.")
try:
    yolo_model = YOLO('best.pt')
    reader = easyocr.Reader(['en'], gpu=True)
except Exception as e:
    print(f"❌ ERROR: Failed to load YOLO or OCR: {e}")
    exit()

print("Loading DistilBERT Weakness Detection Model...")
model_path = "./fyp_trained_model"
try:
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
    distilbert_model = DistilBertForSequenceClassification.from_pretrained(model_path)
    with open(f'{model_path}/label_encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
except Exception as e:
    print(f"❌ ERROR: Failed to load DistilBERT! Check your folder path: {e}")
    exit()

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

if api_key:
    print("✅ DEBUG: API Key found and loaded.")
else:
    print("❌ ERROR: API Key NOT found! Check your .env file.")
    exit()

client = Groq(api_key=api_key)
print(f"✅ All AI Models Successfully Loaded! (Using Groq model: {GROQ_MODEL})\n")


# ==========================================
# SHARED HELPER: ROBUST JSON EXTRACTION
# ==========================================
def extract_json(raw_output, expect="object"):
    """
    Pulls a JSON object {} or array [] out of an LLM response,
    even if there's extra text/markdown fences around it.
    expect: "object" or "array"
    """
    pattern = r'\{.*\}' if expect == "object" else r'\[.*\]'
    match = re.search(pattern, raw_output, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON {expect} found in LLM response.")
    return json.loads(match.group(0))


# ==========================================
# PHASE 2: VISION & EVALUATION FUNCTIONS
# ==========================================

def run_vision_pipeline(image_path):
    print(f"\n--- 🔍 SCANNING IMAGE WITH YOLO: {image_path} ---")

    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ ERROR: Could not read image at '{image_path}'. Check the path/filename.")
        return []

    results = yolo_model.predict(source=image_path, conf=0.20, iou=0.7, verbose=False)

    all_mcqs_text = []

    boxes = results[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0:
        print("⚠️ WARNING: YOLO detected 0 boxes on this image. Check image quality / model confidence threshold.")
        return []

    print("--- 📝 READING TEXT WITH OCR ---")

    # Sort boxes from Top to Bottom based on Y-axis
    boxes = sorted(boxes, key=lambda b: b[1])

    for count, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cropped_piece = img[y1:y2, x1:x2]

        ocr_results = reader.readtext(cropped_piece, detail=1, paragraph=False, mag_ratio=3, text_threshold=0.3, min_size=7)
        ocr_results.sort(key=lambda x: x[0][0][1])

        final_text_list = []
        current_line = []
        line_tolerance = 15

        if len(ocr_results) > 0:
            previous_y = ocr_results[0][0][0][1]
            for (ocr_box, text, confidence) in ocr_results:
                current_y = ocr_box[0][1]
                current_x = ocr_box[0][0]

                if abs(current_y - previous_y) <= line_tolerance:
                    current_line.append((current_x, text))
                else:
                    current_line.sort(key=lambda item: item[0])
                    for item in current_line:
                        final_text_list.append(item[1])
                    current_line = [(current_x, text)]
                    previous_y = current_y

            if current_line:
                current_line.sort(key=lambda item: item[0])
                for item in current_line:
                    final_text_list.append(item[1])

        final_text = " ".join(final_text_list)
        all_mcqs_text.append(final_text)
        print(f"✅ Box {count + 1} Extracted successfully!")

    print("\n========================================")
    print("🎓 FINAL PAPER DATA (FOR GEN AI) 🎓")
    print("========================================")
    for idx, mcq in enumerate(all_mcqs_text):
        print(f"MCQ {idx + 1}: {mcq}")

    return all_mcqs_text


def get_groq_evaluation(all_mcqs_text):
    print("\n========================================")
    print("🤖 AI EXAMINER IS GRADING THE PAPER...")
    print("========================================")

    prompt = f"""
    You are an Expert AI Examiner. Evaluate the following raw OCR text containing an MCQ exam.

    RAW OCR DATA LIST:
    {all_mcqs_text}

    UNIVERSAL RULES:
    1. SCATTERED OCR DATA (CRITICAL): The OCR list is not always in perfect order. It may contain full questions and isolated text snippets. The isolated snippets represent the student's selected answers.
    2. DETECTIVE MAPPING (UNIVERSAL LOGIC): Logically match every isolated snippet to its corresponding question using text similarity and option letters (A, B, C, D). Even if there are OCR typos, link it to the closest matching option.
    3. STRICT FAITHFUL EXTRACTION: Extract exactly what the student selected, even if it has typos. Just autocorrect the typos (not answer) and write it in student_selected.
    4. INCOMPLETE DATA REJECTION: If the OCR text contains ONLY options (no question) or ONLY a question (no options), IGNORE IT completely. Do NOT guess or hallucinate missing text. Do NOT include it in the JSON output.
    5. CORRECT ANSWER: Provide the universally accepted correct answer based on Computer Science domain knowledge.
    6. SMART GRADING: If the student's selection logically matches the correct answer (ignoring minor OCR typos), the status is "Correct" (Marks: 1). Otherwise, "Incorrect" (Marks: 0).

    CRITICAL: Respond ONLY with a valid JSON ARRAY of objects. Do not use markdown blocks like ```json.
    Format exactly like this:
    [
      {{
        "question_no": "Q1",
        "question_text": "Full text of the question here?",
        "student_selected": "The option student picked (or 'Not Detected by OCR')",
        "correct_answer": "The actual correct option",
        "status": "Correct" or "Incorrect",
        "marks": 1 or 0
      }}
    ]
    """

    max_retries = 3
    for attempt in range(max_retries):
        try:
            print(f"⏳ Attempt {attempt + 1}/{max_retries} — Sending request to Groq...")
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )

            raw_output = response.choices[0].message.content
            evaluated_json = extract_json(raw_output, expect="array")

            print("\n--- 🏆 FINAL STUDENT RESULT CARD (JSON) 🏆 ---")
            print(json.dumps(evaluated_json, indent=4))

            return evaluated_json

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} Failed! Error: {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep((attempt + 1) * 2)

    return []


def normalize_answer(raw_answer):
    """
    Cleans up LLM output so 'B', 'Option B', 'b)', 'The answer is B.' etc.
    all collapse to the same comparable value ('B').
    Falls back to the cleaned lowercase text if no single A/B/C/D letter is found
    (useful for non-MCQ / short-answer questions).
    """
    text = raw_answer.strip()

    match = re.search(r'\b([ABCD])\b', text.upper())
    if match:
        return match.group(1)

    cleaned = re.sub(r'[^\w\s]', '', text).strip().lower()
    return cleaned


def check_llm_confidence(question_text, n_attempts=3):
    """
    SUPERVISOR'S REQUIRED CHECK:
    Verifies how reliable the LLM's own "correct_answer" judgement is,
    independent of whether the student got it right or wrong.

    Method: self-consistency. Ask the LLM the same question n_attempts times
    at LOW temperature (this is a factual-recall check, not a creative task -
    high temperature only adds wording noise, not real uncertainty signal).
    Answers are normalized before comparing, so 'B' vs 'Option B' vs 'b)' count
    as the same answer instead of triggering a false mismatch.

    Confidence uses majority vote, not "all 3 must be identical":
    - All 3 agree       -> High
    - 2 out of 3 agree  -> Medium
    - All 3 different   -> Low
    """
    answers_raw = []
    answers_normalized = []

    verify_prompt = f"""You are a CS domain expert examiner. This is a multiple-choice question.
Respond with ONLY the single correct option letter: A, B, C, or D.
Do NOT explain. Do NOT repeat the option text. Output exactly one capital letter and nothing else.

Question: {question_text}
"""

    for i in range(n_attempts):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": verify_prompt}],
                temperature=0.2
            )
            raw_ans = response.choices[0].message.content.strip()
            answers_raw.append(raw_ans)
            answers_normalized.append(normalize_answer(raw_ans))
        except Exception as e:
            answers_raw.append(f"ERROR: {e}")
            answers_normalized.append(f"ERROR_{i}")

    from collections import Counter
    counts = Counter(answers_normalized)
    most_common_answer, most_common_count = counts.most_common(1)[0]

    if most_common_count == n_attempts:
        confidence = "High"
    elif most_common_count >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {
        "confidence": confidence,
        "final_answer": most_common_answer,
        "attempts_raw": answers_raw,
        "attempts_normalized": answers_normalized
    }


def add_confidence_checks(evaluated_json):
    """
    Runs check_llm_confidence on every question in the evaluated result
    and attaches the confidence info to each item.
    Call this AFTER get_groq_evaluation().
    """
    print("\n========================================")
    print("🔬 VERIFYING LLM ANSWER CONFIDENCE...")
    print("========================================")

    for item in evaluated_json:
        q_text = item.get("question_text", "")
        if not q_text:
            item["llm_confidence"] = "Skipped - no question text"
            continue

        result = check_llm_confidence(q_text)
        item["llm_confidence"] = result["confidence"]
        item["llm_confidence_final_answer"] = result["final_answer"]
        item["llm_confidence_attempts_raw"] = result["attempts_raw"]

        flag = "✅" if result["confidence"] == "High" else "⚠️"
        print(f"{flag} {item.get('question_no', '?')} -> Confidence: {result['confidence']}")

    return evaluated_json


def batch_detect_weakness_topics(incorrect_questions):
    """
    Takes a list of question texts and returns a dictionary mapping 
    each question to its classified topic in ONE single API call.
    """
    valid_topics = [
        "Operating Systems", "Database Systems", "Computer Networks", 
        "Web Development", "Artificial Intelligence", "Information Security", 
        "Software Engineering", "Data Structures" , " Object-Oriented Programming" , "Compiler Design", "Cloud Computing", "Functional Programming"
    ]
    
    prompt = f"""
    Classify the following Computer Science questions into EXACTLY ONE topic from the list below.
    Valid Topics: {', '.join(valid_topics)}
    
    Questions List:
    {json.dumps(incorrect_questions, indent=2)}
    
    CRITICAL: Respond ONLY with a valid JSON dictionary where the key is the exact question text, and the value is the exact topic name. Do NOT add markdown blocks.
    """
    
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_output = response.choices[0].message.content
        return extract_json(raw_output, expect="object")
            
    except Exception as e:
        print(f"❌ Batch Classification Error: {e}")
        # Fallback dictionary if API fails
        return {q: "General Computer Science" for q in incorrect_questions}


# ==========================================
# PHASE 3: TUTOR & QUIZ FUNCTIONS
# ==========================================

def get_intent(student_input):
    prompt = f"""
    Student's message: "{student_input}"

    Task: Analyze the intent of the student. Choose strictly ONE category:
    1. SPECIFIC_DOUBT: If the student asks a specific question or explicitly asks for an example (e.g., "give me an example", "what is ReLU?").
    2. EXPLAIN_MORE: If the student simply says they don't understand and need more detail without specifying what.
    3. GENERATE_QUIZ: If the student says they understand, asks for a test, or says move on.

    CRITICAL: Respond ONLY with a valid JSON object. Do NOT add any extra text or markdown formatting.
    Format:
    {{
        "intent": "SPECIFIC_DOUBT" or "EXPLAIN_MORE" or "GENERATE_QUIZ",
        "extracted_doubt": "If they asked a specific question or asked for an example, write it here. Otherwise, leave empty."
    }}
    """
    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_output = response.choices[0].message.content
        return extract_json(raw_output, expect="object")
    except Exception as e:
        print(f"\n[DEBUG: Intent Detection Failed. Defaulting to EXPLAIN_MORE. Error: {e}]")
        return {"intent": "EXPLAIN_MORE", "extracted_doubt": ""}


def teach_topic(topic, attempt, student_message="", failed_concepts_context=""):
    """Teaching function adapted for Micro-Topics with Easy-to-Understand Depth"""

    context_instruction = ""
    if failed_concepts_context:
        context_instruction = f"""
        CRITICAL CONTEXT: The student failed the following specific questions in the domain of '{topic}':
        {failed_concepts_context}

        YOUR TASK: DO NOT teach the broad subject of '{topic}'. Analyze the failed questions, extract the EXACT specific micro-concepts/sub-topics they cover, and teach ONLY those specific weak areas.
        """

    interaction_context = ""
    if student_message:
        interaction_context = f"""
        The student just said this to you: "{student_message}"

        YOUR IMMEDIATE REACTION RULES:
        1. IF ABUSIVE: Tell them gently but firmly to keep it professional.
        2. IF OFF-TOPIC: Remind them to focus on the current topic first.
        3. IF VALID DOUBT: Address their specific confusion using a simple real-world analogy.
        """

    if attempt == 1:
        instructions = "Provide a clean, easy-to-read Concept Map for the specific sub-topics extracted from the failed questions. Use simple headings and bullet points. Explain the core mechanics using everyday analogies (like a restaurant, traffic, filing cabinet, etc.). Do NOT use dense academic jargon. End by asking: 'Does this make sense, or do you have any specific doubts before we start the quiz?'"
    elif attempt == 2:
        instructions = "Address the student's message directly. Explain the 'Why' and 'How' deeply but in VERY SIMPLE English. Break down any complex engineering terms into plain words. End by asking: 'Do you have any MORE doubts, or should we jump into the quiz?'"
    else:
        instructions = "The student failed the mock quiz. Provide a FINAL, definitive explanation using a VERY SIMPLE REAL-WORLD SYSTEM ARCHITECTURE as an example (e.g., how this is used in an everyday app like WhatsApp, an ATM, or an online store). Focus exclusively on fixing their mistakes. IMPORTANT: Do NOT ask any questions at the end. Conclude with an encouraging remark."

    prompt = f"""
    You are a highly skilled but very friendly Senior Software Engineer mentoring a final-semester Computer Science student.
    Broad Domain: {topic}
    {context_instruction}
    {interaction_context}

    YOUR TEACHING STYLE & LANGUAGE RULES (CRITICAL):
    1. EXTREME SIMPLICITY: You MUST write in basic, conversational, everyday English. Speak as if you are explaining a concept to a junior developer over a cup of coffee.
    2. DEEP BUT ACCESSIBLE: Explain the core technical mechanics deeply, but WITHOUT using overly complex academic jargon. 
    3. BAN POST-GRAD JARGON: Do NOT use extreme terminology unless absolutely necessary. If you must use a technical term, you MUST instantly explain it in plain words.
    4. USE ANALOGIES: Always use relatable, real-world analogies.
    5. VISUAL STRUCTURE: Keep paragraphs very short. Use simple bullet points. Do not create massive, intimidating tables.

    Instructions: {instructions}

    CRITICAL RULE: Respond COMPLETELY in simple English. Maintain your friendly, mentor-like persona. Never sound like a strict academic textbook.
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"\n[DEBUG: teach_topic Groq call failed: {e}]")
        return "⚠️ Tutor is temporarily unavailable (API error). Please try again."

def generate_mock_quiz(topic, failed_concepts_context=""):
    """Generates 10 Advanced Questions based strictly on what was just taught"""

    prompt = f"""
    You are an expert university examiner. Generate EXACTLY 10 ADVANCED multiple-choice questions (MCQs).

    Context of what the student just learned/failed:
    {failed_concepts_context}

    RULES:
    1. Base the questions STRICTLY on the specific micro-topics mentioned in the context above, falling under the broad domain of '{topic}'.
    2. The questions MUST be at an advanced university level (scenario-based, code-logic based, or deep conceptual), matching the depth of your recent teaching.
    3. CRITICAL RANDOMIZATION: You MUST heavily randomize the 'correct_option' across 'A', 'B', 'C', and 'D'. Do NOT favor 'A' or 'B'.

    CRITICAL: Respond ONLY with a valid JSON ARRAY of objects. Do NOT add any extra text.
    Format exactly like this:
    [
        {{
            "question": "Advanced Question text here?",
            "A": "Option A",
            "B": "Option B",
            "C": "Option C",
            "D": "Option D",
            "correct_option": "C",
            "explanation": "Deep technical reason why this is correct"
        }}
    ]
    """

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        raw_output = response.choices[0].message.content
        return extract_json(raw_output, expect="array")
    except Exception as e:
        print(f"\n[DEBUG: Quiz Generation Failed. Executing Fallback. Error: {e}]")
        return [{"question": "System Error: Cannot load custom questions.", "A": "Ok", "B": "Error", "C": "Skip", "D": "None", "correct_option": "B", "explanation": "API failed to generate JSON array."}]


# ==========================================
# PHASE 4: THE MASTER EXECUTION (INTEGRATION)
# ==========================================
if __name__ == "__main__":

    test_image = 'T5.jpeg'  # Apni image ka naam lagao

    raw_mcqs = run_vision_pipeline(test_image)
    if not raw_mcqs:
        print("❌ ERROR: No text found in the image by OCR!")
        exit()

    evaluated_data = get_groq_evaluation(raw_mcqs)
    if not evaluated_data:
        print("❌ ERROR: Failed to get valid evaluation JSON from Groq.")
        exit()

    # ---- NEW: Supervisor's required LLM accuracy/confidence check ----
    evaluated_data = add_confidence_checks(evaluated_data)

    # ... (OCR and Evaluation logic here) ...

    print("\n--- 🔍 DETECTING MICRO-WEAKNESSES (BATCH MODE) ---")
    weak_topics_dict = {}
    
    # Step 1: Sirf un questions ko filter karo jo galat hain
    incorrect_questions = [
        item.get("question_text", "") 
        for item in evaluated_data 
        if str(item.get("status", "")).strip().lower() == "incorrect" and item.get("question_text")
    ]
    
    if incorrect_questions:
        # Step 2: Ek hi call mein saare topics mangwa lo
        topic_mapping = batch_detect_weakness_topics(incorrect_questions)
        
        # Step 3: Apne data ko update karo
        for item in evaluated_data:
            q_text = item.get("question_text", "")
            if q_text in topic_mapping:
                topic = topic_mapping[q_text]
                item["weakness_topic"] = topic
                print(f"🚨 WEAKNESS DETECTED -> Question: {item.get('question_no', '?')} | Topic: {topic}")
                
                if topic not in weak_topics_dict:
                    weak_topics_dict[topic] = []
                weak_topics_dict[topic].append(q_text)

    output_filename = 'final_student_report.json'
    with open(output_filename, 'w') as f:
        json.dump(evaluated_data, f, indent=4)
    print(f"✅ Result saved to '{output_filename}'. (Includes llm_confidence field per question.)")

    if not weak_topics_dict:
        print("\n🏆 Student passed all questions. No weakness detected. Session Complete.")
        exit()

    print(f"\n🧠 Initializing Tutor for {len(weak_topics_dict)} unique subjects...")
    time.sleep(1)

    for current_topic, failed_qs_list in weak_topics_dict.items():
        print(f"\n=======================================================")
        print(f"📚 --- STARTING TEACHING MODULE: {current_topic.upper()} --- 📚")
        print(f"=======================================================")

        original_mistakes_str = "\n".join([f"- {q}" for q in failed_qs_list])

        tutor_reply = teach_topic(current_topic, attempt=1, failed_concepts_context=original_mistakes_str)
        print(f"\n🤖 Tutor:\n{tutor_reply}")

        current_context_for_quiz = original_mistakes_str

        while True:
            student_msg = input("\n👨‍🎓 Student (Your turn): ")
            intent_data = get_intent(student_msg)
            user_intent = intent_data.get("intent")

            if user_intent == "GENERATE_QUIZ":
                print("\n✅ Tutor: Alright, let's test your advanced understanding... (Generating 10 questions, please wait ⏳)")
                break

            else:
                print(f"\n⏳ Tutor is thinking deeply...")
                tutor_reply = teach_topic(current_topic, attempt=2, student_message=student_msg, failed_concepts_context=original_mistakes_str)
                print(f"\n🤖 Tutor:\n{tutor_reply}")

        print("\n📝 --- MOCK QUIZ TIME (10 ADVANCED QUESTIONS) --- 📝")
        quiz_data_list = generate_mock_quiz(current_topic, failed_concepts_context=current_context_for_quiz)
        total_questions = len(quiz_data_list)
        score = 0
        current_quiz_mistakes = []

        for index, q in enumerate(quiz_data_list, start=1):
            print(f"\n--- Question {index}/{total_questions} ---")
            print(f"Q: {q.get('question', 'N/A')}")
            print(f"A: {q.get('A', 'N/A')}")
            print(f"B: {q.get('B', 'N/A')}")
            print(f"C: {q.get('C', 'N/A')}")
            print(f"D: {q.get('D', 'N/A')}")

            student_ans = input("Choose the correct option (A/B/C/D): ").strip().upper()
            while student_ans not in ['A', 'B', 'C', 'D']:
                student_ans = input("Invalid input! Please enter exactly A, B, C, or D: ").strip().upper()

            if student_ans == q.get('correct_option'):
                print("✅ Correct!")
                score += 1
            else:
                print(f"❌ Wrong! The correct answer was: {q.get('correct_option', 'N/A')}")
                print(f"💡 Explanation: {q.get('explanation', 'N/A')}")
                current_quiz_mistakes.append(f"Failed Question: {q.get('question', '')} | Concept missed: {q.get('explanation', '')}")

        print(f"\n📊 --- QUIZ RESULT FOR {current_topic.upper()} --- 📊")
        print(f"Your Score: {score}/{total_questions}")

        if len(current_quiz_mistakes) == 0:
            print(f"🏆 Tutor: Excellent work! You answered everything correctly. The micro-topics for '{current_topic}' are fully cleared!")
        else:
            print(f"⚠️ Tutor: You still struggled with some advanced concepts. Let me give you a final breakdown of your mistakes before we move on.\n")

            failed_concepts_history = "\n".join(current_quiz_mistakes)
            print(f"⏳ Tutor is preparing your final advanced architectural feedback...")
            final_remedy = teach_topic(current_topic, attempt=3, failed_concepts_context=failed_concepts_history)

            print(f"\n🤖 Final Tutor Note:\n{final_remedy}")
            print(f"\n🚨 Tutor: I have cleared your advanced misconceptions. We are now closing '{current_topic}' and moving on.")

    print("\n🏁 --- ALL WEAKNESSES ADDRESSED. SESSION END --- 🏁")