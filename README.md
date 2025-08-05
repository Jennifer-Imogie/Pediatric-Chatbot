---
title: Pediatric Pulmonology Chatbot
emoji: 🫁
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.40.0
app_file: app.py
pinned: false
license: mit
---

# Pediatric Pulmonology Chatbot

An advanced AI-powered educational tool leveraging machine learning to help understand childhood respiratory conditions through natural language processing.

## Purpose

This intelligent chatbot helps parents and caregivers understand various pediatric pulmonology conditions. It uses state-of-the-art machine learning models to interpret natural language descriptions of symptoms and provide comprehensive educational information about respiratory conditions in children.

## AI-Powered Features

### Advanced Natural Language Understanding
- Bio-Clinical BERT Integration: Medical domain-specific language model for accurate symptom interpretation
- TF-IDF Similarity Matching: Finds the most relevant conditions based on symptom descriptions
- Multi-layered Classification: Combines ML models with enhanced rule-based systems
- Confidence Scoring: Provides reliability indicators for each prediction
- Flexible Input Processing: Understands various ways of describing symptoms

### Smart Response Generation
- Context-Aware Analysis: Considers multiple symptoms and their relationships
- Confidence-Based Responses: Adapts advice based on prediction certainty
- Educational Content Delivery: Structured medical information presentation
- Progressive Disclosure: Asks for more details when needed for better accuracy

## Medical Conditions Covered

### Common Pediatric Respiratory Conditions
- Asthma - Chronic airway inflammation and bronchospasm
- Bronchiolitis - Viral infection affecting small airways in infants
- Pneumonia - Bacterial or viral lung infection
- Chronic Cough - Persistent cough lasting >4 weeks

### Specialized Pulmonology Conditions
- Paradoxical Vocal Fold Movement (PVFM) - Vocal cord dysfunction
- Subglottic Stenosis - Airway narrowing below vocal cords
- Acute Respiratory Distress Syndrome (ARDS) - Severe lung inflammation
- Tracheoesophageal Fistula - Abnormal connection between airways
- Laryngeal Web - Congenital vocal cord membrane
- Primary Ciliary Dyskinesia - Genetic ciliary dysfunction
- Pulmonary Arterial Hypertension - Elevated lung blood pressure
- Hereditary Hemorrhagic Telangiectasia - Genetic vascular disorder
- Esophageal Atresia - Congenital esophageal malformation
- Asbestosis - Occupational lung disease (rare in children)

## Advanced Capabilities

### Natural Language Processing
With rule-based logic: "My child wheezes" (exact match required)
With ML model: "Kid sounds funny when breathing and gets tired easily"

### Intelligent Symptom Analysis
- Multi-symptom Recognition: Processes complex symptom combinations
- Age-Appropriate Assessment: Considers the child's developmental stage
- Severity Indicators: Identifies urgent vs. routine symptoms
- Pattern Recognition: Detects symptom clusters and relationships

### Confidence-Based Responses
- High Confidence (>60%): Detailed condition information
- Moderate Confidence (30-60%): Condition info with caveats  
- Low Confidence (<30%): Requests more specific information

## How to Use Effectively
### How to Run
Try the chatbot live on [Hugging Face Spaces](https://huggingface.co/spaces/imogie/Pediatric_Chatbot)

### Optimal Input Examples:
- Good: "My 3-year-old has been wheezing at night and coughs when running around"
- Better: "Baby has stuffy nose, breathing fast, and won't finish bottles for 3 days"
- Best: "Toddler making high-pitched breathing sounds, gets tired easily, voice sounds different"

### Information to Include:
- Child's age (infant, toddler, school-age)
- Specific symptoms (wheezing, cough, fever, breathing changes)
- Duration and timing of symptoms
- Triggers or patterns you've noticed
- Impact on daily activities (feeding, sleep, play)

## Technical Architecture

### Machine Learning Stack
- Primary Model: Bio-Clinical BERT (emilyalsentzer/Bio_ClinicalBERT)
- Fallback Model: PubMed BERT (microsoft/BiomedNLP-PubMedBERT)
- Similarity Engine: TF-IDF with Cosine Similarity
- Enhanced Rules: Multi-keyword scoring system
- Framework: Hugging Face Transformers + scikit-learn

## Performance Metrics

### Model Capabilities
- **Language Flexibility**: Handles 90%+ of natural symptom descriptions
- **Accuracy**: High precision for well-described symptoms
- **Coverage**: 14 major pediatric pulmonology conditions
- **Response Time**: <3 seconds average processing time
- **Confidence Calibration**: Reliable uncertainty quantification

### Supported Input Variations
- **Incomplete sentences**: "baby breathing funny"
- **Medical terms**: "stridor, tachypnea, dyspnea"
- **Lay descriptions**: "sounds wheezy", "can't catch breath"
- **Multiple symptoms**: Complex symptom combinations
- **Contextual details**: Age, triggers, duration

## Medical Disclaimer & Safety

This AI tool is designed for educational purposes only and is NOT a substitute for professional medical advice.

### Always Consult Healthcare Providers For:
- **Diagnosis and Treatment**: Professional medical evaluation required
- **Medication Decisions**: Prescription and dosing guidance
- **Urgent Symptoms**: Any concerning changes in child's condition
- **Follow-up Care**: Monitoring and management plans

### Seek Immediate Emergency Care For:
- Severe breathing difficulty or inability to breathe
- Blue coloration of lips, face, or fingernails (cyanosis)
- Loss of consciousness or extreme lethargy
- Inability to speak or cry due to breathing problems
- High fever with severe breathing distress

## Technical Resources

### Medical Knowledge Sources

Data gathered from the following data source 
- Mayo Clinic — https://www.mayoclinic.org
- PubMed Central (PMC) — https://pmc.ncbi.nlm.nih.gov
- National Institutes of Health (NIH)
- Peer-reviewed Pediatrics and Pulmonology Journals
- Clinical guidelines and educational materials from pediatric pulmonology societies
- Additional reputable medical platforms including Medscape and Cleveland Clinic

### Model Documentation 
- **Bio-Clinical BERT**: Specialized medical language understanding
- **TF-IDF Similarity**: Symptom pattern matching algorithms
- **Confidence Scoring**: Uncertainty quantification methods
- **Response Templates**: Structured medical information delivery


## Development Team
Team 3 - Pediatric Pulmonology Chatbot Project
- Leslie El
- Jennifer Imogie
- Barakat Abubakar

---


