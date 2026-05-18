# AI Learning Companion: Academic Weakness Mitigation & Progress Tracking System

AI Learning Companion is an automated academic assessment and personalized tutoring pipeline. While it seamlessly tracks student progress by evaluating exam papers, its core objective is to identify and overcome specific academic weaknesses. It achieves this by providing targeted learning assistance, clearing conceptual doubts, and generating dynamic mock quizzes to reinforce learning.

**Core Technical Pipeline:**
* **Computer Vision (YOLOv8):** Utilizes a custom-trained YOLOv8 object detection model to accurately detect, localize, and crop individual Multiple Choice Questions (MCQs) from raw images of exam papers.
* **Text Extraction (OCR):** Processes the cropped regions to extract raw text with high accuracy.
* **NLP Evaluation (DistilBERT):** Employs a fine-tuned DistilBERT transformer model to analyze the parsed text, verify answers, and map incorrect responses to specific curriculum weaknesses.
* **Remedial Tutoring & Quiz Generation:** Automatically generates customized mock quizzes and provides AI-driven doubt-clearing sessions focused exclusively on the student's identified weak areas.

**Key Features:**

* **Automated Assessment:** Seamless MCQ evaluation and grading directly from raw exam images.
* **Dynamic Weakness Detection:** Intelligently maps incorrect answers to specific curriculum topics to pinpoint knowledge gaps.
* **Personalized Doubt Clearing:** AI-driven remedial tutoring tailored to clear specific conceptual misunderstandings.
* **Adaptive Mock Quizzes:** Automatic generation of targeted quizzes focused exclusively on the student's identified weak areas for reinforcement.
* **Seamless Web Integration:** Structured output generation (JSON) ready for integration with frontend web applications.
* **Scalable Data Pipeline:** Built-in generation scripts for continuous dataset expansion and model improvement.