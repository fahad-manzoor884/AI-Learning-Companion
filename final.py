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
    print(f"✅ DEBUG: API Key found! First 8 characters: {api_key[:8]}...")
else:
    print("❌ ERROR: API Key NOT found! Check your .env file.")
    exit()

client = Groq(api_key=api_key)
print("✅ All AI Models Successfully Loaded!\n")


# ==========================================
# PHASE 2: VISION & EVALUATION FUNCTIONS
# ==========================================

def run_vision_pipeline(image_path):
    print(f"\n--- 🔍 SCANNING IMAGE WITH YOLO: {image_path} ---")
    img = cv2.imread(image_path)
    results = yolo_model.predict(source=image_path, conf=0.30, verbose=False)

    all_mcqs_text = []

    print("--- 📝 READING TEXT WITH OCR ---")
    boxes = results[0].boxes.xyxy.cpu().numpy()
    
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
    4. MISSING ANSWERS: If no answer is found for a question in the list, write "Not Detected by OCR". If no option is given for a question, then do not write it in JSON also If only options are written and the question is not written then also do not write it in JSON.
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
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            raw_output = response.choices[0].message.content
            clean_json_str = re.sub(r"```json\n|\n```", "", raw_output).strip()
            evaluated_json = json.loads(clean_json_str)

            print("\n--- 🏆 FINAL STUDENT RESULT CARD (JSON) 🏆 ---")
            print(json.dumps(evaluated_json, indent=4))
            
            return evaluated_json

        except Exception as e:
            print(f"❌ Attempt {attempt + 1} Failed! Error: {e}")
            if attempt == max_retries - 1:
                return []
            time.sleep((attempt + 1) * 2)
            
    return []

def detect_weakness_topic(question_text):
    inputs = tokenizer(question_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = distilbert_model(**inputs)
        predicted_class_id = torch.argmax(outputs.logits, dim=1).item()
    return encoder.inverse_transform([predicted_class_id])[0]


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
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0
        )
        raw_output = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return {"intent": "EXPLAIN_MORE", "extracted_doubt": ""}
    except Exception as e:
        print(f"\n[DEBUG: Intent Detection Failed. Defaulting to EXPLAIN_MORE. Error: {e}]")
        return {"intent": "EXPLAIN_MORE", "extracted_doubt": ""}

def teach_topic(topic, attempt, student_message="", failed_concepts_context=""):
    """Teaching function adapted for Micro-Topics and Advanced University-Level Depth"""
    
    # YAHAN MAGIC HAI: Hum LLM ko directly ghalat sawal bhej rahay hain taake wo chota topic pakray
    context_instruction = ""
    if failed_concepts_context:
        context_instruction = f"""
        CRITICAL CONTEXT: The student failed the following specific questions in the domain of '{topic}':
        {failed_concepts_context}
        
        YOUR TASK: DO NOT teach the broad subject of '{topic}'. Analyze the failed questions, extract the EXACT specific micro-concepts/sub-topics they cover (e.g., if the question is about 'ReLU', teach ONLY Activation Functions, not the whole Neural Network history), and teach ONLY those specific weak areas.
        """

    interaction_context = ""
    if student_message:
        interaction_context = f"""
        The student just said this to you: "{student_message}"
        
        YOUR IMMEDIATE REACTION RULES:
        1. IF ABUSIVE: If the student uses profanity or disrespects you, SCOLD THEM HARSHLY like a strict professor.
        2. IF OFF-TOPIC: Firmly tell them to focus on the current topic first.
        3. IF VALID DOUBT: Address their specific confusion deeply.
        """

    if attempt == 1:
        instructions = "Provide a highly structured, ADVANCED UNIVERSITY-LEVEL Concept Map for the specific sub-topics extracted from the failed questions. Use headings, bullet points, and include deep technical mechanics. Do not give basic definitions. End by asking: 'Do you have any specific doubts, or are you ready for the advanced quiz?'"
    elif attempt == 2:
        instructions = "Address the student's message directly. Provide a HIGHLY DETAILED, ADVANCED EXPLANATION. Explain the 'Why' and 'How' at an engineering level (including architecture, edge cases, or mathematical intuition if applicable) so they are prepared for a difficult exam. End by asking: 'Do you have any MORE doubts, or should I generate the quiz now?'"
    else: 
        instructions = "The student failed the mock quiz. Provide a FINAL, definitive explanation using an ADVANCED REAL-WORLD SYSTEM ARCHITECTURE as an example (e.g., how this sub-topic is actually used in production software). Focus exclusively on fixing their advanced mistakes. IMPORTANT: Do NOT ask any questions at the end. Conclude firmly."

    prompt = f"""
    You are an expert, highly strict, and brilliant University Computer Science Professor teaching final-year engineering students.
    Broad Domain: {topic}
    {context_instruction}
    {interaction_context}
    
    Instructions: {instructions}
    
    CRITICAL RULE: Respond COMPLETELY in English. Teach at an ADVANCED level. Maintain your strict professor persona at all times.
    """
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.6 
    )
    return response.choices[0].message.content

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
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4 
        )
        raw_output = response.choices[0].message.content
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        raise Exception("LLM did not return any JSON Array.")
    except Exception as e:
        print(f"\n[DEBUG: Quiz Generation Failed. Executing Fallback. Error: {e}]")
        return [{"question": f"System Error: Cannot load custom questions.", "A": "Ok", "B": "Error", "C": "Skip", "D": "None", "correct_option": "B", "explanation": "API failed to generate JSON array."}]
    

# ==========================================
# PHASE 4: THE MASTER EXECUTION (INTEGRATION)
# ==========================================
if __name__ == "__main__":
    
    test_image = 'T3.jpeg' # Apni image ka naam lagao
    
    raw_mcqs = run_vision_pipeline(test_image)
    if not raw_mcqs:
        print("❌ ERROR: No text found in the image by OCR!")
        exit()
        
    evaluated_data = get_groq_evaluation(raw_mcqs)
    if not evaluated_data:
        print("❌ ERROR: Failed to get valid evaluation JSON from Groq.")
        exit()
        
    print("\n--- 🔍 DETECTING MICRO-WEAKNESSES ---")
    # YAHAN CHANGE AYA HAI: Set ki jagah Dictionary taake Sawal bhi save hon
    weak_topics_dict = {} 
    
    for item in evaluated_data:
        topic = detect_weakness_topic(item.get("question_text", ""))
        item["weakness_topic"] = topic
        
        if str(item.get("status", "")).strip().lower() == "incorrect":
            print(f"🚨 WEAKNESS DETECTED -> Question: {item['question_no']} | Topic: {topic}")
            if topic not in weak_topics_dict:
                weak_topics_dict[topic] = []
            # Ghalat sawal ko dictionary mein save kar lo
            weak_topics_dict[topic].append(item.get("question_text", ""))

    output_filename = 'final_student_report.json'
    with open(output_filename, 'w') as f:
        json.dump(evaluated_data, f, indent=4)
    print(f"✅ Result saved to '{output_filename}'.")

    if not weak_topics_dict:
        print("\n🏆 Student passed all questions. No weakness detected. Session Complete.")
        exit()
        
    print(f"\n🧠 Initializing Tutor for {len(weak_topics_dict)} unique subjects...")
    time.sleep(1)

    # Loop through Dictionary (Subject and its specific failed questions)
    for current_topic, failed_qs_list in weak_topics_dict.items():
        print(f"\n=======================================================")
        print(f"📚 --- STARTING TEACHING MODULE: {current_topic.upper()} --- 📚")
        print(f"=======================================================")
        
        # Original failed questions ko ek string mein convert karo
        original_mistakes_str = "\n".join([f"- {q}" for q in failed_qs_list])
        
        # 1. INITIAL ADVANCED TEACHING PHASE
        tutor_reply = teach_topic(current_topic, attempt=1, failed_concepts_context=original_mistakes_str)
        print(f"\n🤖 Tutor:\n{tutor_reply}")
        
        # 2. OPEN-ENDED DOUBT CLEARING PHASE
        current_context_for_quiz = original_mistakes_str # Default context for quiz
        
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

        # 3. ONE-SHOT MOCK QUIZ PHASE (Advanced)
        print("\n📝 --- MOCK QUIZ TIME (10 ADVANCED QUESTIONS) --- 📝")
        quiz_data_list = generate_mock_quiz(current_topic, failed_concepts_context=current_context_for_quiz)
        total_questions = len(quiz_data_list)
        score = 0
        current_quiz_mistakes = []
        
        for index, q in enumerate(quiz_data_list, start=1):
            print(f"\n--- Question {index}/{total_questions} ---")
            print(f"Q: {q['question']}")
            print(f"A: {q['A']}")
            print(f"B: {q['B']}")
            print(f"C: {q['C']}")
            print(f"D: {q['D']}")
            
            student_ans = input("Choose the correct option (A/B/C/D): ").strip().upper()
            while student_ans not in ['A', 'B', 'C', 'D']:
                student_ans = input("Invalid input! Please enter exactly A, B, C, or D: ").strip().upper()
            
            if student_ans == q['correct_option']:
                print("✅ Correct!")
                score += 1
            else:
                print(f"❌ Wrong! The correct answer was: {q['correct_option']}")
                print(f"💡 Explanation: {q['explanation']}")
                current_quiz_mistakes.append(f"Failed Question: {q['question']} | Concept missed: {q['explanation']}")
        
        # 4. FINAL EVALUATION & OPTION B REMEDY
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