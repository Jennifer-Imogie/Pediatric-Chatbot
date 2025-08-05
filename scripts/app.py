import gradio as gr
import spacy
import pandas as pd
import numpy as np
import re
from datetime import datetime
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import warnings
warnings.filterwarnings('ignore')

# Initialize the chatbot class
class PediatricPulmonologyChatbot:
    def __init__(self):
        # Initialize ML models
        self.setup_ml_models()
        
        # Training data for intent classification
        self.training_data = {
            "asthma": [
                "my child is wheezing", "he has a tight chest and can't breathe", "child sounds breathless",
                "she's coughing and short of breath", "wheezing at night", "tight chest", "inhaler needed",
                "difficulty breathing", "whistling sound when breathing", "chest tightness during exercise",
                "chronic wheezing", "allergic asthma", "exercise induced asthma", "nocturnal coughing"
            ],
            "bronchiolitis": [
                "baby has stuffy nose and cough", "infant wheezing with fever", "rapid breathing in baby",
                "congested and breathing fast", "nasal flaring", "chest congestion in baby", "rsv symptoms",
                "baby struggling to feed", "fast shallow breathing", "infant with cold symptoms",
                "runny nose and wheezing baby", "difficulty feeding baby", "baby sounds congested"
            ],
            "pneumonia": [
                "child has chest pain and fever", "coughing up mucus", "high fever and fatigue",
                "breathing fast with chills", "chest crackling sounds", "very tired with fever",
                "productive cough", "bacterial infection lungs", "viral pneumonia", "chest x-ray abnormal",
                "difficulty breathing with fever", "shaking chills", "pleuritic chest pain"
            ],
            "chronic cough": [
                "persistent cough for weeks", "dry cough won't stop", "cough worsens at night",
                "ongoing cough after cold", "chronic dry cough", "cough for more than month",
                "post-infectious cough", "habitual cough", "psychogenic cough", "cough variant asthma",
                "lingering cough", "cough without other symptoms", "nighttime coughing fits"
            ],
            "paradoxical vocal fold movement": [
                "stridor when inhaling", "tight throat can't breathe in", "voice disappears suddenly",
                "throat closes during breathing", "panic with breathing", "exercise induced stridor",
                "vocal cord dysfunction", "throat tightness stress", "difficulty inhaling only",
                "feels like choking", "throat spasm", "can't get air in", "inspiratory stridor"
            ],
            "subglottic stenosis": [
                "high pitched breathing sound", "noisy breathing", "stridor at rest",
                "airway narrowing", "difficulty breathing lying down", "hoarse voice chronic",
                "breathing obstruction", "harsh breathing sounds", "airway surgery needed",
                "congenital stridor", "biphasic stridor", "progressive breathing difficulty"
            ],
            "acute respiratory distress syndrome": [
                "severe breathing difficulty", "needs ventilator", "lung inflammation severe",
                "icu admission breathing", "oxygen not helping", "rapid onset breathing failure",
                "bilateral lung infiltrates", "severe hypoxia", "mechanical ventilation needed",
                "acute lung injury", "sepsis related breathing", "shock lung"
            ],
            "hereditary hemorrhagic telangiectasia": [
                "frequent nosebleeds", "family history bleeding", "abnormal blood vessels",
                "chronic bleeding", "telangiectasia", "arteriovenous malformation",
                "genetic bleeding disorder", "recurrent epistaxis", "vascular malformation",
                "hereditary bleeding", "osler weber rendu", "pulmonary avm"
            ],
            "tracheoesophageal fistula": [
                "coughs when feeding", "milk comes through nose", "choking during feeding",
                "aspiration with feeds", "connection windpipe foodpipe", "surgical repair needed",
                "esophageal atresia", "feeding difficulties newborn", "recurrent pneumonia feeding",
                "congenital anomaly", "h-type fistula", "feeding intolerance"
            ],
            "laryngeal web": [
                "weak hoarse cry", "congenital voice problems", "stridor since birth",
                "membrane vocal cords", "difficulty intubating", "web larynx", "glottic web",
                "voice abnormality birth", "inspiratory stridor newborn", "cry abnormal",
                "congenital laryngeal anomaly", "airway obstruction birth"
            ],
            "primary ciliary dyskinesia": [
                "chronic wet cough", "sinus infections recurrent", "ear infections frequent",
                "immotile cilia", "kartagener syndrome", "bronchiectasis", "daily sputum",
                "chronic rhinosinusitis", "hearing loss chronic", "situs inversus",
                "chronic respiratory infections", "ciliary dysfunction", "genetic lung disease"
            ],
            "pulmonary arterial hypertension": [
                "fatigue with exertion", "shortness breath exercise", "fainting spells",
                "high lung pressure", "right heart failure", "blue lips exercise",
                "syncope exertion", "chest pain exertion", "elevated pulmonary pressure",
                "right heart strain", "eisenmenger syndrome", "pah diagnosis"
            ],
            "esophageal atresia": [
                "frothy saliva newborn", "cannot swallow", "choking feeding attempts",
                "milk regurgitation", "unable pass feeding tube", "congenital esophagus",
                "feeding tube won't pass", "excessive drooling baby", "aspiration feeding",
                "surgical correction needed", "type c ea", "blind pouch esophagus"
            ],
            "asbestosis": [
                "asbestos exposure history", "lung scarring", "occupational lung disease",
                "interstitial fibrosis", "progressive breathlessness", "chest imaging abnormal",
                "environmental exposure", "lung function decline", "pulmonary fibrosis",
                "occupational hazard", "mesothelioma risk", "pleural plaques"
            ]
        }
        
        # Setup TF-IDF vectorizer for similarity matching
        self.setup_similarity_model()
        # Knowledge base from your original code
        self.knowledge_base = {
            "asthma": {
                "definition": "Asthma is a chronic condition that causes inflammation and narrowing of the airways, leading to wheezing, breathlessness, and coughing.",
                "symptoms": [
                    "wheezing",
                    "shortness of breath", 
                    "coughing, especially at night or early morning",
                    "tightness in the chest"
                ],
                "red_flags": [
                    "severe difficulty breathing",
                    "lips turning blue",
                    "child unable to speak or cry",
                    "no improvement with inhaler"
                ],
                "advice": "Use a prescribed inhaler, keep the child in an upright position, avoid triggers like dust or pollen, and seek emergency care if symptoms worsen."
            },
            "bronchiolitis": {
                "definition": "Bronchiolitis is a common lung infection in infants and young children, usually caused by a virus, that leads to inflammation and congestion in the small airways.",
                "symptoms": [
                    "cough",
                    "runny nose", 
                    "wheezing",
                    "fast or shallow breathing",
                    "poor feeding"
                ],
                "red_flags": [
                    "grunting or flaring nostrils while breathing",
                    "difficulty feeding or drinking",
                    "chest retractions",
                    "cyanosis (bluish skin)"
                ],
                "advice": "Keep the child well hydrated, monitor for worsening symptoms, and seek medical attention if breathing becomes labored or feeding decreases."
            },
            "pneumonia": {
                "definition": "Pneumonia is an infection of the lungs that causes the air sacs to fill with fluid or pus, leading to cough, fever, and difficulty breathing.",
                "symptoms": [
                    "fever",
                    "cough with phlegm",
                    "chest pain", 
                    "rapid breathing",
                    "fatigue"
                ],
                "red_flags": [
                    "very high fever",
                    "confusion or lethargy",
                    "labored breathing",
                    "cyanosis"
                ],
                "advice": "Ensure the child rests, drinks plenty of fluids, and consult a doctor. Severe symptoms may require antibiotics or hospitalization."
            },
            "chronic cough": {
                "definition": "Chronic cough is a cough that lasts more than 4 weeks in children. It can be dry or productive and may indicate an underlying condition.",
                "symptoms": [
                    "persistent cough for more than 4 weeks",
                    "hoarseness",
                    "dry or wet cough",
                    "cough worsens at night or with exercise"
                ],
                "red_flags": [
                    "cough with blood",
                    "weight loss", 
                    "difficulty breathing",
                    "loss of appetite"
                ],
                "advice": "Avoid environmental irritants, keep the child hydrated, and seek medical evaluation to determine the underlying cause."
            },
            "paradoxical vocal fold movement": {
                "definition": "PVFM is a condition in which the vocal folds close when they should open during breathing, often triggered by stress or irritants.",
                "symptoms": [
                    "stridor",
                    "sudden shortness of breath",
                    "tightness in the throat", 
                    "difficulty inhaling"
                ],
                "red_flags": [
                    "sudden and total voice loss",
                    "stridor during both inhale and exhale",
                    "severe anxiety with breathing difficulty"
                ],
                "advice": "Encourage relaxed throat breathing, avoid triggers, and work with a speech-language pathologist for breathing retraining."
            },
            "subglottic stenosis": {
                "definition": "Subglottic stenosis is a narrowing of the airway just below the vocal cords, which can be congenital or acquired.",
                "symptoms": [
                    "noisy breathing (stridor)",
                    "difficulty breathing during activity",
                    "voice changes or hoarseness"
                ],
                "red_flags": [
                    "severe breathing difficulty",
                    "cyanosis (bluish skin or lips)",
                    "stridor at rest"
                ],
                "advice": "Avoid irritants, monitor breathing, and seek evaluation by an ENT specialist."
            },
            "acute respiratory distress syndrome": {
                "definition": "ARDS is a severe inflammatory reaction in the lungs causing fluid accumulation and difficulty in oxygen exchange.",
                "symptoms": [
                    "rapid breathing",
                    "shortness of breath", 
                    "low oxygen levels"
                ],
                "red_flags": [
                    "extreme difficulty breathing",
                    "requires mechanical ventilation",
                    "persistent hypoxia"
                ],
                "advice": "Requires ICU admission and oxygen support. Early recognition and treatment are crucial."
            },
            "hereditary hemorrhagic telangiectasia": {
                "definition": "HHT is a genetic disorder causing abnormal blood vessel formation, leading to bleeding in organs like lungs and brain.",
                "symptoms": [
                    "frequent nosebleeds",
                    "shortness of breath",
                    "unexplained anemia"
                ],
                "red_flags": [
                    "stroke-like symptoms",
                    "brain or lung hemorrhage", 
                    "significant hemoptysis (coughing blood)"
                ],
                "advice": "Genetic counseling, monitor for bleeding, and treat complications promptly."
            },
            "tracheoesophageal fistula": {
                "definition": "A TEF is an abnormal connection between the trachea and esophagus, often congenital.",
                "symptoms": [
                    "coughing or choking during feeding",
                    "recurrent respiratory infections",
                    "difficulty swallowing"
                ],
                "red_flags": [
                    "cyanosis while feeding",
                    "aspiration pneumonia",
                    "failure to thrive"
                ],
                "advice": "Requires surgical correction. Ensure safe feeding methods until repaired."
            },
            "laryngeal web": {
                "definition": "Laryngeal web is a congenital or acquired membrane that partially obstructs the vocal cords.",
                "symptoms": [
                    "weak or hoarse cry",
                    "stridor",
                    "breathing difficulty during exertion"
                ],
                "red_flags": [
                    "airway obstruction",
                    "progressive stridor",
                    "poor weight gain due to effort in breathing"
                ],
                "advice": "ENT evaluation for surgical intervention. Avoid airway irritants."
            },
            "primary ciliary dyskinesia": {
                "definition": "PCD is a rare genetic disorder where cilia in the lungs do not function properly, leading to mucus build-up and infections.",
                "symptoms": [
                    "chronic wet cough",
                    "nasal congestion",
                    "recurrent ear and sinus infections"
                ],
                "red_flags": [
                    "bronchiectasis",
                    "hearing loss",
                    "progressive lung damage"
                ],
                "advice": "Airway clearance therapies, regular monitoring, and genetic counseling."
            },
            "pulmonary arterial hypertension": {
                "definition": "PAH is increased blood pressure in the arteries of the lungs, making it harder for the heart to pump blood.",
                "symptoms": [
                    "fatigue",
                    "shortness of breath during exertion",
                    "fainting spells"
                ],
                "red_flags": [
                    "cyanosis",
                    "chest pain",
                    "syncope (fainting)"
                ],
                "advice": "Specialist care with medications to reduce pressure. Avoid strenuous activity."
            },
            "esophageal atresia": {
                "definition": "Esophageal atresia is a birth defect where the esophagus does not connect to the stomach.",
                "symptoms": [
                    "frothy saliva",
                    "difficulty feeding",
                    "choking or coughing when feeding"
                ],
                "red_flags": [
                    "aspiration pneumonia",
                    "cyanosis during feeding",
                    "inability to pass a feeding tube"
                ],
                "advice": "Requires urgent surgical correction. Supportive care until surgery."
            },
            "asbestosis": {
                "definition": "Asbestosis is a chronic lung disease caused by inhaling asbestos fibers, rare in children unless exposed.",
                "symptoms": [
                    "persistent dry cough",
                    "chest tightness",
                    "shortness of breath"
                ],
                "red_flags": [
                    "respiratory failure",
                    "clubbing of fingers",
                    "cor pulmonale"
                ],
                "advice": "Prevent exposure, monitor lung function, and seek pulmonary care."
            }
        }
        
        
        # Try to load the spaCy model - fallback to rule-based if not available
        self.nlp = None
        try:
            # You can add your trained spaCy model here later
            pass
        except:
            print("Using ML-enhanced classification")

    def setup_ml_models(self):
        """Initialize machine learning models"""
        try:
            # Use a medical/clinical BERT model for better understanding
            print("Loading ML models...")
            self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
            self.model = AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=14)
            print("Bio-Clinical BERT loaded successfully")
        except Exception as e:
            print(f"Could not load Bio-Clinical BERT: {e}")
            try:
                # Fallback to general medical model
                self.classifier = pipeline(
                    "text-classification",
                    model="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
                    return_all_scores=True
                )
                print("PubMed BERT loaded successfully")
            except Exception as e2:
                print(f"Could not load PubMed BERT: {e2}")
                # Final fallback to general model
                try:
                    self.classifier = pipeline(
                        "text-classification", 
                        model="distilbert-base-uncased",
                        return_all_scores=True
                    )
                    print("DistilBERT loaded as fallback")
                except:
                    print("Using rule-based classification only")
                    self.classifier = None

    def setup_similarity_model(self):
        """Setup TF-IDF similarity model for intent matching"""
        # Prepare all training texts and labels
        all_texts = []
        self.labels = []
        
        for condition, examples in self.training_data.items():
            for example in examples:
                all_texts.append(example.lower())
                self.labels.append(condition)
        
        # Create TF-IDF vectors
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(all_texts)
        print(f"TF-IDF model trained on {len(all_texts)} examples")

    def predict_condition_ml(self, text):
        """Use ML models to predict condition"""
        text_lower = text.lower()
        
        # Method 1: TF-IDF Similarity
        try:
            query_vector = self.vectorizer.transform([text_lower])
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top 3 most similar examples
            top_indices = similarities.argsort()[-3:][::-1]
            top_similarities = similarities[top_indices]
            
            if top_similarities[0] > 0.1:  # Minimum similarity threshold
                predicted_condition = self.labels[top_indices[0]]
                confidence = float(top_similarities[0])
                
                # Calculate condition-level confidence by averaging similar examples
                condition_scores = {}
                for idx in top_indices:
                    if similarities[idx] > 0.05:
                        condition = self.labels[idx]
                        if condition not in condition_scores:
                            condition_scores[condition] = []
                        condition_scores[condition].append(similarities[idx])
                
                # Average scores for each condition
                for condition in condition_scores:
                    condition_scores[condition] = np.mean(condition_scores[condition])
                
                best_condition = max(condition_scores, key=condition_scores.get)
                best_confidence = condition_scores[best_condition]
                
                return best_condition, float(best_confidence)
        except Exception as e:
            print(f"TF-IDF prediction error: {e}")
        
        # Method 2: Rule-based fallback with enhanced patterns
        return self.rule_based_classifier_enhanced(text_lower)

    def rule_based_classifier_enhanced(self, text):
        """Enhanced rule-based classification with better pattern matching"""
        text_lower = text.lower()
        
        # Create scoring system for multiple keywords
        condition_scores = {}
        
        # Enhanced keyword patterns
        patterns = {
            "asthma": [
                (["wheez", "wheezing"], 0.8),
                (["tight chest", "chest tight"], 0.7),
                (["shortness", "breath"], 0.6),
                (["inhaler"], 0.9),
                (["asthma"], 1.0),
                (["breathless"], 0.6),
                (["night", "cough"], 0.5),
                (["exercise", "breathing"], 0.4)
            ],
            "bronchiolitis": [
                (["baby", "infant"], 0.6),
                (["runny nose", "stuffy"], 0.5),
                (["fast breathing", "rapid"], 0.7),
                (["bronchiolitis"], 1.0),
                (["rsv"], 0.8),
                (["nasal flaring"], 0.8),
                (["feeding", "difficulty"], 0.6)
            ],
            "pneumonia": [
                (["fever", "high"], 0.6),
                (["chest pain"], 0.7),
                (["mucus", "phlegm"], 0.6),
                (["pneumonia"], 1.0),
                (["chills"], 0.5),
                (["fatigue", "tired"], 0.4),
                (["crackling"], 0.8)
            ],
            "chronic cough": [
                (["persistent", "chronic"], 0.7),
                (["weeks", "month"], 0.6),
                (["dry cough"], 0.6),
                (["night", "cough"], 0.5),
                (["4 weeks", "four weeks"], 0.8),
                (["ongoing"], 0.5)
            ],
            "paradoxical vocal fold movement": [
                (["stridor"], 0.9),
                (["throat", "tight"], 0.7),
                (["voice", "loss"], 0.6),
                (["inhaling", "difficult"], 0.7),
                (["vocal cord"], 0.8),
                (["choking", "feeling"], 0.6)
            ]
            # Add more patterns for other conditions...
        }
        
        # Score each condition
        for condition, pattern_list in patterns.items():
            score = 0
            for keywords, weight in pattern_list:
                if isinstance(keywords, list):
                    if all(keyword in text_lower for keyword in keywords):
                        score += weight
                else:
                    if keywords in text_lower:
                        score += weight
            
            if score > 0:
                condition_scores[condition] = score
        
        if condition_scores:
            best_condition = max(condition_scores, key=condition_scores.get)
            confidence = min(condition_scores[best_condition] / 2.0, 0.95)  # Normalize
            return best_condition, confidence
        
        return None, 0.0

    def rule_based_classifier(self, text):
        """Rule-based classification as fallback"""
        text_lower = text.lower()
        
        # Asthma indicators
        if any(word in text_lower for word in ["wheez", "tight chest", "shortness of breath", "inhaler", "asthma"]):
            return "asthma"
        
        # Bronchiolitis indicators  
        elif any(word in text_lower for word in ["baby", "infant", "runny nose", "fast breathing", "bronchiolitis"]):
            return "bronchiolitis"
            
        # Pneumonia indicators
        elif any(word in text_lower for word in ["fever", "chest pain", "mucus", "pneumonia", "chills"]):
            return "pneumonia"
            
        # Chronic cough indicators
        elif any(word in text_lower for word in ["chronic cough", "persistent cough", "4 weeks", "long cough"]):
            return "chronic cough"
            
        # PVFM indicators
        elif any(word in text_lower for word in ["stridor", "throat tight", "voice loss", "inhaling difficult"]):
            return "paradoxical vocal fold movement"
            
        # Add more conditions...
        elif any(word in text_lower for word in ["noisy breathing", "stenosis", "airway narrow"]):
            return "subglottic stenosis"
            
        elif any(word in text_lower for word in ["nosebleed", "bleeding", "telangiectasia"]):
            return "hereditary hemorrhagic telangiectasia"
            
        elif any(word in text_lower for word in ["choking feeding", "tracheoesophageal", "fistula"]):
            return "tracheoesophageal fistula"
            
        elif any(word in text_lower for word in ["hoarse cry", "laryngeal web", "weak voice"]):
            return "laryngeal web"
            
        elif any(word in text_lower for word in ["wet cough chronic", "ciliary", "sinus infection"]):
            return "primary ciliary dyskinesia"
            
        elif any(word in text_lower for word in ["pulmonary hypertension", "fainting", "blue lips"]):
            return "pulmonary arterial hypertension"
            
        elif any(word in text_lower for word in ["feeding difficult", "esophageal atresia", "choking milk"]):
            return "esophageal atresia"
            
        elif any(word in text_lower for word in ["asbestos", "lung scar", "occupational"]):
            return "asbestosis"
            
        return None

    def generate_response(self, user_input, history):
        """Generate response based on user input using ML models"""
        if not user_input or not user_input.strip():
            return "Please describe your child's symptoms so I can help you better."
        
        # Use ML model for classification
        condition, confidence = self.predict_condition_ml(user_input)
        
        # Set confidence threshold
        confidence_threshold = 0.15
        
        if not condition or confidence < confidence_threshold:
            return f"""I need more specific information to help you better. Please describe:

• **Specific symptoms** (e.g., "wheezing", "fever", "difficulty breathing")
• **How long** has your child had these symptoms?
• **Your child's age** (infant, toddler, school-age)
• **When symptoms occur** (night, during activity, after eating)
• **Any triggers** you've noticed

**Example:** "My 3-year-old has been wheezing at night and coughs when running"

**Remember:** This is for educational purposes only. Always consult a pediatrician for proper medical evaluation.

**Current analysis:** Based on your input "{user_input}", I detected some patterns but need more details for accurate information (confidence: {confidence:.2f})."""

        # Get information from knowledge base
        info = self.knowledge_base.get(condition)
        if not info:
            return f"I identified this might be related to **{condition.replace('_', ' ').title()}** (confidence: {confidence:.2f}), but I need more information to provide specific guidance. Please consult a pediatric pulmonologist."

        # Build comprehensive response with confidence indicator
        confidence_indicator = "High" if confidence > 0.6 else "Moderate" if confidence > 0.3 else "Low"
        
        response = f"## **Analysis Result**\n\n"
        response += f"**Possible Condition:** {condition.replace('_', ' ').title()}\n"
        response += f"**Confidence Level:** {confidence_indicator} ({confidence:.2f})\n\n"
        
        response += f"**Definition:**\n{info['definition']}\n\n"
        
        if info['symptoms']:
            response += "**Common Symptoms:**\n"
            for symptom in info['symptoms'][:4]:
                response += f"• {symptom}\n"
            response += "\n"
        
        if info['red_flags']:
            response += "**RED FLAGS - Seek URGENT Medical Care if you notice:**\n"
            for flag in info['red_flags']:
                response += f"•{flag}\n"
            response += "\n"
        
        if info['advice']:
            response += f"**General Advice:**\n{info['advice']}\n\n"
        
        # Add confidence-based disclaimer
        if confidence < 0.4:
            response += "**Low Confidence Notice:**\nThis prediction has lower confidence. Please provide more specific symptoms or consult a healthcare provider for accurate assessment.\n\n"
        
        response += """---
**IMPORTANT DISCLAIMER:**
- This is for educational purposes only
- Always consult a pediatrician for proper diagnosis  
- If symptoms worsen or red flags appear, seek immediate medical attention
- Call emergency services if your child has severe breathing difficulty"""

        return response

# Initialize chatbot
chatbot = PediatricPulmonologyChatbot()

def chat_function(message, history):
    """Main chat function for Gradio"""
    if not message or not message.strip():
        return "", history
    
    try:
        response = chatbot.generate_response(message, history)
    except Exception as e:
        response = f"I encountered an error processing your request. Please try rephrasing your question or consult a healthcare provider. Error: {str(e)}"
    
    history.append([message, response])
    return "", history

def clear_conversation():
    """Clear chat history"""
    return [], ""

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Pediatric Pulmonology Assistant",
    css="""
    .gradio-container {
        max-width: 900px;
        margin: 0 auto;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    """
) as demo:
    
    # Header
    gr.Markdown("""
    # 🫁 Pediatric Pulmonology Assistant
    
    **AI-Powered Support for Understanding Childhood Respiratory Conditions**
    
    This assistant can help you understand various pediatric pulmonology conditions including asthma, bronchiolitis, pneumonia, and other respiratory disorders in children.
    
    ---
    """)
    
    # Warning Box
    gr.HTML("""
    <div class="warning-box">
        <h3> Medical Disclaimer</h3>
        <p><strong>This tool is for educational purposes only and is not a substitute for professional medical advice.</strong> 
        Always consult with a qualified pediatrician or healthcare provider for proper diagnosis and treatment. 
        In case of emergency or severe symptoms, call emergency services immediately.</p>
    </div>
    """)
    
    # Main Interface
    with gr.Row():
        with gr.Column():
            chatbot_interface = gr.Chatbot(
                height=500,
                show_label=False
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Describe your child's symptoms (e.g., 'My 3-year-old has been wheezing and coughing at night')",
                    container=False,
                    scale=4,
                    lines=2
                )
                send_btn = gr.Button("Send", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                
    # Quick Examples
    gr.Markdown("""
    Example Questions You Can Ask:
    
    - "My 2-year-old has been wheezing and coughing, especially at night"
    - "Baby has runny nose, fast breathing, and won't eat well" 
    - "Child has persistent cough for 6 weeks after a cold"
    - "My child makes a high-pitched sound when breathing in"
    - "Toddler has fever, chest pain, and is breathing fast"
    """)
    
    # Conditions Covered
    with gr.Accordion("Conditions This Assistant Can Help With", open=False):
        gr.Markdown("""
        **Common Conditions:**
        - Asthma
        - Bronchiolitis  
        - Pneumonia
        - Chronic Cough
        
        **Specialized Conditions:**
        - Paradoxical Vocal Fold Movement (PVFM)
        - Subglottic Stenosis
        - Acute Respiratory Distress Syndrome (ARDS)
        - Tracheoesophageal Fistula
        - Laryngeal Web
        - Primary Ciliary Dyskinesia
        - Pulmonary Arterial Hypertension
        - Hereditary Hemorrhagic Telangiectasia
        - Esophageal Atresia
        - Asbestosis (rare in children)
        """)
    
    # Event handlers
    msg.submit(chat_function, inputs=[msg, chatbot_interface], outputs=[msg, chatbot_interface])
    send_btn.click(chat_function, inputs=[msg, chatbot_interface], outputs=[msg, chatbot_interface])
    clear_btn.click(clear_conversation, outputs=[chatbot_interface, msg])

# Launch the app
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
