---
title: Pediatric Pulmonology Assistant
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# Pediatric Pulmonology Assistant

An AI-powered educational tool to help understand childhood respiratory conditions.

## Purpose

This chatbot assists parents and caregivers in understanding various pediatric pulmonology conditions. It provides educational information about symptoms, red flags, and general advice for respiratory conditions in children.

## Conditions Covered

### Common Conditions
- **Asthma** - Chronic airway inflammation
- **Bronchiolitis** - Viral infection in small airways  
- **Pneumonia** - Lung infection
- **Chronic Cough** - Persistent cough lasting >4 weeks

### Specialized Conditions
- **Paradoxical Vocal Fold Movement (PVFM)**
- **Subglottic Stenosis** 
- **Acute Respiratory Distress Syndrome (ARDS)**
- **Tracheoesophageal Fistula**
- **Laryngeal Web**
- **Primary Ciliary Dyskinesia**
- **Pulmonary Arterial Hypertension**
- **Hereditary Hemorrhagic Telangiectasia**
- **Esophageal Atresia**
- **Asbestosis** (rare in children)

## Features

- **Symptom Analysis**: Describes symptoms based on user input
- **Red Flags**: Identifies warning signs requiring urgent care
- **Educational Content**: Provides definitions and explanations
- **General Advice**: Offers basic guidance and next steps
- **User-Friendly Interface**: Clean, medical-themed design

## How to Use

1. **Describe Symptoms**: Enter your child's symptoms in natural language
2. **Get Information**: Receive educational content about possible conditions
3. **Understand Red Flags**: Learn when to seek urgent medical care
4. **Follow Advice**: Get general guidance for next steps

### Example Inputs:
- "My 3-year-old has been wheezing and coughing at night"
- "Baby has runny nose and fast breathing"
- "Child has persistent cough for 6 weeks"
- "Toddler makes high-pitched sound when breathing"

## Important Medical Disclaimer

**This tool is for educational purposes only and is NOT a substitute for professional medical advice.**

- Always consult a qualified pediatrician for proper diagnosis
- Seek immediate medical attention for severe symptoms
- Call emergency services for breathing emergencies
- Do not delay medical care based on this tool's suggestions

## Technical Details

### Built With
- **Gradio** - Web interface framework
- **Python** - Core programming language
- **spaCy** - Natural language processing (when available)
- **Rule-based Classification** - Fallback symptom analysis

### Model Architecture
- **Knowledge Base**: Comprehensive medical information database
- **Rule-based Classifier**: Pattern matching for symptom recognition
- **Confidence Scoring**: Reliability assessment for responses

## Target Audience

- **Parents & Caregivers**: Understanding children's respiratory symptoms
- **Medical Students**: Learning pediatric pulmonology basics

## Accuracy & Limitations

### Strengths
- Comprehensive condition coverage
- Evidence-based medical information
- Clear red flag identification
- Educational focus

### Limitations
- Cannot replace professional diagnosis
- Rule-based classification (not ML-trained)
- Limited to covered conditions
- No personalized medical advice

## Emergency Resources

**Always call emergency services immediately if a child has:**
- Severe difficulty breathing
- Blue lips or face
- Loss of consciousness
- Cannot speak due to breathing difficulty


**Developed by Team 3 - Pediatric Pulmonology Project**
- Leslie EL
- Jennifer Imogie
- Barakat Abubakar

*Remember: When in doubt, always consult a healthcare professional!*
