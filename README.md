# AI Learning Companion: Intelligent Student Progress Tracking System

AI Learning Companion is an automated academic assessment and progress tracking pipeline designed to evaluate student performance seamlessly. By leveraging a hybrid architecture of Computer Vision and Natural Language Processing (NLP), this system eliminates manual grading and provides deep insights into a student's academic weaknesses.

# Core Technical Pipeline:

Computer Vision (YOLOv8): Utilizes a custom-trained YOLOv8 object detection model to accurately detect, localize, and crop individual Multiple Choice Questions (MCQs) from raw images of exam papers.

Text Extraction (OCR): Processes the cropped regions to extract raw text with high accuracy.

NLP Evaluation (DistilBERT): Employs a fine-tuned DistilBERT transformer model to analyze the parsed text, verify the student's selected answers against the correct options, and intelligently categorize the question into specific domain topics (e.g., Computer Networks, Deep Learning).

# Key Features:

Automated MCQ evaluation and grading.

Dynamic weakness detection that maps incorrect answers to specific curriculum topics.

Structured output generation (JSON) for seamless integration with frontend web applications.

Scalable data generation scripts for continuous dataset building and model improvement.