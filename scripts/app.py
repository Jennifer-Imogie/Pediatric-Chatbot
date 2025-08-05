import gradio as gr
import spacy
import pandas as pd
import numpy as np
import re
from datetime import datetime
import os

# Initialize the chatbot class
class PediatricPulmonologyChatbot:
    def __init__(self):
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
            # In production, you might want to download and include your trained model
            # For now, we'll use rule-based classification
            pass
        except:
            print("Using rule-based classification as fallback")

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
        """Generate response based on user input"""
        if not user_input or not user_input.strip():
            return "Please describe your child's symptoms so I can help you better."
        
        # Classify the condition
        condition = self.rule_based_classifier(user_input)
        
        if not condition:
            return """I couldn't identify a specific condition from your description. Please provide more details about:

• **Specific symptoms** (cough, fever, breathing difficulty, etc.)
• **Duration** of symptoms
• **Child's age**
• **Any triggers** you've noticed

**Remember:** This is for informational purposes only. Always consult with a pediatrician for proper medical evaluation."""

        # Get information from knowledge base
        info = self.knowledge_base.get(condition)
        if not info:
            return f"I identified this might be related to **{condition.replace('_', ' ').title()}**, but I need more information to provide specific guidance. Please consult a pediatric pulmonologist."

        # Build comprehensive response
        response = f"## 🩺 **Possible Condition: {condition.replace('_', ' ').title()}**\n\n"
        
        response += f"**📋 Definition:**\n{info['definition']}\n\n"
        
        if info['symptoms']:
            response += "**🔍 Common Symptoms:**\n"
            for symptom in info['symptoms'][:4]:  # Show top 4 symptoms
                response += f"• {symptom}\n"
            response += "\n"
        
        if info['red_flags']:
            response += "**🚨 RED FLAGS - Seek URGENT Medical Care if you notice:**\n"
            for flag in info['red_flags']:
                response += f"• ⚠️ {flag}\n"
            response += "\n"
        
        if info['advice']:
            response += f"**💡 General Advice:**\n{info['advice']}\n\n"
        
        response += """---
**⚠️ IMPORTANT DISCLAIMER:**
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
        <h3>⚠️ Medical Disclaimer</h3>
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
                placeholder="👋 Hello! I'm here to help you understand pediatric respiratory conditions. Describe your child's symptoms and I'll provide relevant information.",
                bubble_full_width=False,
                avatar_images=("👨‍👩‍👧‍👦", "🩺"),
                show_copy_button=True
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Describe your child's symptoms (e.g., 'My 3-year-old has been wheezing and coughing at night')",
                    container=False,
                    scale=4,
                    lines=2
                )
                send_btn = gr.Button("Send 📤", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
                
    # Quick Examples
    gr.Markdown("""
    ### 💡 Example Questions You Can Ask:
    
    - "My 2-year-old has been wheezing and coughing, especially at night"
    - "Baby has runny nose, fast breathing, and won't eat well" 
    - "Child has persistent cough for 6 weeks after a cold"
    - "My child makes a high-pitched sound when breathing in"
    - "Toddler has fever, chest pain, and is breathing fast"
    """)
    
    # Conditions Covered
    with gr.Accordion("🔍 Conditions This Assistant Can Help With", open=False):
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
