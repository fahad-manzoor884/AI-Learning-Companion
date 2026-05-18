import os
import time
import csv
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)

# 🎓 CS Degree ke lag bhag tamam ahem subjects
topics = [
    "Programming Fundamentals", "Object Oriented Programming", "Data Structures", 
    "Design and Analysis of Algorithms", "Database Systems", "Operating Systems", 
    "Computer Networks", "Software Engineering", "Web Development", 
    "Artificial Intelligence", "Machine Learning", "Deep Learning",
    "Parallel and Distributed Computing", "Graph Algorithms", "Compiler Construction",
    "Theory of Automata", "Digital Logic Design", "Computer Architecture",
    "Information Security", "Cryptography", "Blockchain", 
    "Numerical Computing", "Human Computer Interaction", "Cloud Computing",
    "Mobile App Development", "Data Science", "Internet of Things"
]

# Total sawal = 4 batches * 25 = 100 per topic (Total ~2700 Questions)
batches_per_topic = 4
questions_per_batch = 25

csv_filename = "ultimate_cs_weakness_dataset.csv"

with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(["Question", "Topic"])

    for topic in topics:
        print(f"\n🚀 Topic: {topic} ke sawalat generate ho rahe hain...")
        for batch in range(batches_per_topic):
            prompt = f"""
            You are a Computer Science professor. Generate {questions_per_batch} distinct, university-level exam questions strictly about '{topic}'.
            Provide ONLY the question text. Do NOT provide options (A, B, C, D). Do NOT provide answers. Do NOT number the questions.
            Each question must be on a new line. Do not write any intro or outro text.
            """
            try:
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7 
                )
                
                raw_text = response.choices[0].message.content.strip()
                questions = [q.strip() for q in raw_text.split('\n') if q.strip()]
                
                for q in questions:
                    writer.writerow([q, topic])
                    
                print(f"  ✅ Batch {batch + 1} complete. ({len(questions)} questions added)")
                time.sleep(3) 
                
            except Exception as e:
                print(f"  ❌ Error in {topic} batch {batch + 1}: {e}")
                time.sleep(10)

print(f"\n🎉 Dataset successfully saved to {csv_filename}!")