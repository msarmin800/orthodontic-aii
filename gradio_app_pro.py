import gradio as gr
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

print("Loading models...")

# ============= LOAD MODELS =============
all_models = pickle.load(open('all_models.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
best_idx = pickle.load(open('best_model_idx.pkl', 'rb'))

df_for_defaults = pd.read_csv('data.csv')
cooperation_map = {'Good': 2, 'Moderate': 1, 'Poor': 0}
df_for_defaults['Cooperation_Encoded'] = df_for_defaults['Cooperation_Level'].map(cooperation_map)

avg_cooperation = df_for_defaults['Cooperation_Encoded'].mean()
avg_missed = df_for_defaults['Appointments_Skipped'].mean()

names = ['Linear Regression', 'Lasso', 'ElasticNet', 'Random Forest', 'AdaBoost', 'XGBoost']

best_mae = None
with open('model_performance_report.txt', 'r', encoding='utf-8') as f:
    report_text = f.read()
    for line in report_text.split('\n'):
        if 'Mean Absolute Error: ±' in line:
            best_mae = float(line.split('±')[1].split(' ')[0])
            break

predict_folder = 'predict'
if not os.path.exists(predict_folder):
    os.makedirs(predict_folder)

# Feature names
feature_names = ['Age_Start', 'Sex_Encoded', 'Treatment_Extract', 
                'Crowding_Upper', 'Crowding_Lower', 'Overjet', 'ANB', 
                'Cooperation_Encoded', 'Appointments_Skipped']

print("✅ Models loaded!")

# ============= FUNCTIONS =============
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def load_patients_db():
    if os.path.exists('patients_db.json'):
        with open('patients_db.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_patients_db(db):
    with open('patients_db.json', 'w', encoding='utf-8') as f:
        json.dump(db, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

def create_prediction_card(patient_id, patient_name, age, sex, treatment, 
                           prediction, confidence_low, confidence_high, best_model):
    """ایجاد کارت حرفه‌ای"""
    
    img = Image.new('RGB', (1200, 700), color=(245, 247, 250))
    draw = ImageDraw.Draw(img)
    
    # رنگ‌ها
    primary = (52, 152, 219)
    success = (46, 204, 113)
    text = (44, 62, 80)
    light = (189, 195, 199)
    
    try:
        title_font = ImageFont.truetype("arial.ttf", 48)
        heading_font = ImageFont.truetype("arial.ttf", 32)
        name_font = ImageFont.truetype("arial.ttf", 28)
        label_font = ImageFont.truetype("arial.ttf", 20)
        value_font = ImageFont.truetype("arial.ttf", 24)
    except:
        title_font = heading_font = name_font = label_font = value_font = ImageFont.load_default()
    
    y = 20
    
    # Header
    draw.rectangle([(0, 0), (1200, 100)], fill=primary)
    draw.text((60, 30), "🦷 TREATMENT PREDICTION", fill=(255, 255, 255), font=title_font)
    
    # Patient Card
    y = 130
    draw.rectangle([(40, y), (1160, y+120)], outline=text, width=2, fill=(255, 255, 255))
    y += 20
    
    draw.text((60, y), f"Patient: {patient_name}", fill=text, font=name_font)
    y += 40
    draw.text((60, y), f"ID: {patient_id}  |  Age: {age}  |  Sex: {sex}  |  Treatment: {treatment}", 
              fill=text, font=label_font)
    
    # Main Result
    y = 280
    draw.rectangle([(40, y), (1160, y+200)], outline=success, width=4, fill=(240, 255, 245))
    
    y += 30
    draw.text((60, y), "ESTIMATED TREATMENT DURATION", fill=success, font=heading_font)
    
    y += 60
    duration_text = f"{prediction:.1f}"
    draw.text((80, y), duration_text, fill=success, font=ImageFont.truetype("arial.ttf", 80) if 'arial' in str(title_font) else title_font)
    draw.text((450, y+20), "MONTHS", fill=text, font=heading_font)
    
    y += 100
    draw.text((80, y), f"Confidence Range: {confidence_low:.1f} - {confidence_high:.1f} months", 
              fill=text, font=label_font)
    draw.text((700, y), f"Best Model: {best_model}", fill=primary, font=label_font)
    
    # Footer
    y = 650
    draw.text((60, y), f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Accuracy: ±{best_mae:.1f} months", 
              fill=light, font=label_font)
    
    return img

def create_comparison_chart(predictions, patient_name):
    """ایجاد نمودار مقایسه"""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#f5f7fa')
    
    models = list(predictions.keys())
    values = list(predictions.values())
    colors = ['#2ecc71' if i == best_idx else '#3498db' for i in range(len(models))]
    
    bars = ax.bar(models, values, color=colors, edgecolor='#2c3e50', linewidth=2.5, alpha=0.85)
    
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{value:.1f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold', color='#2c3e50')
    
    ax.set_ylabel('Duration (months)', fontsize=12, fontweight='bold')
    ax.set_title(f'All Models Comparison - {patient_name}', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    
    return fig

def predict_patient(patient_id, patient_name, age, sex, treatment, crowding_up, crowding_low, overjet, anb):
    """New Patient Prediction"""
    
    sex_enc = 1 if sex == 'Male' else 0
    treat_enc = 1 if treatment == 'Extract' else 0
    
    # FIX: استفاده از DataFrame با feature names
    X_new_df = pd.DataFrame([[age, sex_enc, treat_enc, crowding_up, crowding_low, 
                              overjet, anb, avg_cooperation, avg_missed]], 
                            columns=feature_names)
    
    X_new_scaled = scaler.transform(X_new_df)
    
    predictions = {}
    for model, name in zip(all_models, names):
        pred = float(model.predict(X_new_scaled)[0])
        predictions[name] = pred
    
    best_pred = predictions[names[best_idx]]
    
    # Create patient folder
    patient_folder = os.path.join(predict_folder, patient_id)
    if not os.path.exists(patient_folder):
        os.makedirs(patient_folder)
    
    # Save to DB
    patients_db = load_patients_db()
    patients_db[patient_id] = {
        'name': patient_name,
        'age': int(age),
        'sex': sex,
        'treatment': treatment,
        'initial_prediction': float(best_pred),
        'created_date': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'all_predictions': predictions
    }
    save_patients_db(patients_db)
    
    # Generate card image
    card_img = create_prediction_card(
        patient_id, patient_name, age, sex, treatment,
        best_pred, best_pred-best_mae, best_pred+best_mae,
        names[best_idx]
    )
    
    # Save card
    card_path = os.path.join(patient_folder, f'{patient_id}_card.png')
    card_img.save(card_path)
    
    # Generate chart
    fig = create_comparison_chart(predictions, patient_name)
    chart_path = os.path.join(patient_folder, f'{patient_id}_chart.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#f5f7fa')
    plt.close()
    
    # Detailed text output
    details = f"""
📋 PATIENT DETAILS
┌──────────────────────────────────────────────────────────────────┐
│ Name: {patient_name}
│ Age: {age} | Sex: {sex}
│ Treatment: {treatment}
│ Measurements: Crowding {crowding_up}/{crowding_low}mm, Overjet {overjet}mm, ANB {anb}°
└──────────────────────────────────────────────────────────────────┘

🎯 INITIAL PREDICTION (with Average Cooperation)

ALL 6 MODEL PREDICTIONS:
"""
    
    for i, (name, pred) in enumerate(predictions.items(), 1):
        marker = "✓ BEST" if name == names[best_idx] else ""
        details += f"{i}. {name}: {pred:.1f} months {marker}\n"
    
    details += f"""
✅ Files saved to: predict/{patient_id}/
📊 Card: {patient_id}_card.png
📈 Chart: {patient_id}_chart.png

Next: Update with cooperation data to get final prediction
"""
    
    return card_img, fig, details

def update_patient(patient_id, cooperation, missed_apt):
    """Update Patient with Cooperation Data"""
    
    patients_db = load_patients_db()
    
    if patient_id not in patients_db:
        return None, None, "❌ Patient not found!"
    
    patient_data = patients_db[patient_id]
    
    sex_enc = 1 if patient_data['sex'] == 'Male' else 0
    treat_enc = 1 if patient_data['treatment'] == 'Extract' else 0
    
    coop_map = {'Good': 2, 'Moderate': 1, 'Poor': 0}
    cooperation_enc = coop_map[cooperation]
    
    # FIX: استفاده از DataFrame با feature names
    X_updated_df = pd.DataFrame([[
        patient_data['age'], sex_enc, treat_enc, 5.0, 5.0, 3.0, 3.0,
        cooperation_enc, missed_apt
    ]], columns=feature_names)
    
    X_updated_scaled = scaler.transform(X_updated_df)
    
    updated_predictions = {}
    for model, name in zip(all_models, names):
        pred = float(model.predict(X_updated_scaled)[0])
        updated_predictions[name] = pred
    
    best_updated = updated_predictions[names[best_idx]]
    
    # Update DB
    patients_db[patient_id]['cooperation'] = cooperation
    patients_db[patient_id]['missed_appointments'] = int(missed_apt)
    patients_db[patient_id]['final_prediction'] = float(best_updated)
    patients_db[patient_id]['updated_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_patients_db(patients_db)
    
    # Generate updated card
    card_img = create_prediction_card(
        patient_id, patient_data['name'], patient_data['age'], patient_data['sex'],
        patient_data['treatment'], best_updated, best_updated-best_mae, best_updated+best_mae,
        names[best_idx]
    )
    
    # Save card
    patient_folder = os.path.join(predict_folder, patient_id)
    card_path = os.path.join(patient_folder, f'{patient_id}_card_final.png')
    card_img.save(card_path)
    
    # Generate chart
    fig = create_comparison_chart(updated_predictions, patient_data['name'])
    chart_path = os.path.join(patient_folder, f'{patient_id}_chart_final.png')
    fig.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='#f5f7fa')
    plt.close()
    
    # Details
    details = f"""
✨ PATIENT UPDATE COMPLETED

📊 COMPARISON:
Initial Prediction: {patient_data['initial_prediction']:.1f} months
Updated Prediction: {best_updated:.1f} months
Change: {best_updated - patient_data['initial_prediction']:+.1f} months

👨‍⚕️ OBSERVED DATA:
Cooperation: {cooperation}
Missed Appointments: {missed_apt}

🎯 UPDATED PREDICTIONS FROM ALL 6 MODELS:
"""
    
    for i, (name, pred) in enumerate(updated_predictions.items(), 1):
        initial = patient_data['all_predictions'][name]
        change = pred - initial
        marker = "✓ BEST" if name == names[best_idx] else ""
        details += f"{i}. {name}: {pred:.1f}m (from {initial:.1f}m, change: {change:+.1f}m) {marker}\n"
    
    details += f"""
✅ Files saved to: predict/{patient_id}/
📊 Card: {patient_id}_card_final.png
📈 Chart: {patient_id}_chart_final.png
"""
    
    return card_img, fig, details

def get_history_table():
    """View Patient History as Table"""
    
    patients_db = load_patients_db()
    
    if not patients_db:
        return pd.DataFrame()
    
    data = []
    for pid, pdata in patients_db.items():
        initial = pdata.get('initial_prediction', '—')
        final = pdata.get('final_prediction', '—')
        
        initial_str = f"{initial:.1f}" if initial != '—' else '—'
        final_str = f"{final:.1f}" if final != '—' else '—'
        
        data.append({
            'ID': pid,
            'Name': pdata['name'],
            'Age': pdata['age'],
            'Sex': pdata['sex'],
            'Initial (months)': initial_str,
            'Final (months)': final_str,
            'Cooperation': pdata.get('cooperation', '—'),
            'Date': pdata.get('created_date', '—')
        })
    
    return pd.DataFrame(data)

# ============= GRADIO INTERFACE =============
with gr.Blocks(
    title="Orthodontic AI",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="green"),
) as demo:
    
    gr.Markdown("""
    # 🦷 Orthodontic AI Prediction System
    **Professional AI-Powered Treatment Duration Prediction**
    """)
    
    with gr.Tabs():
        
        # ============= TAB 1: NEW PATIENT =============
        with gr.Tab("🏥 New Patient Prediction"):
            gr.Markdown("### Create a new prediction for a patient")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### 👤 Patient Information")
                    patient_id_new = gr.Textbox(label="Patient ID", placeholder="P001")
                    patient_name_new = gr.Textbox(label="Patient Name", placeholder="Ahmad")
                    age_new = gr.Slider(5, 80, value=15, label="Age")
                    sex_new = gr.Radio(["Male", "Female"], value="Male", label="Sex")
                    treatment_new = gr.Radio(["Extract", "Non-extract"], value="Extract", label="Treatment")
                
                with gr.Column():
                    gr.Markdown("#### 📐 Clinical Measurements")
                    crowding_up_new = gr.Slider(0, 15, value=5.0, label="Crowding Upper (mm)")
                    crowding_low_new = gr.Slider(0, 15, value=5.0, label="Crowding Lower (mm)")
                    overjet_new = gr.Slider(0, 15, value=3.0, label="Overjet (mm)")
                    anb_new = gr.Slider(-10, 15, value=3.0, label="ANB (degrees)")
            
            predict_btn = gr.Button("🚀 Generate Prediction", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    card_output = gr.Image(label="📊 Prediction Card")
                with gr.Column():
                    chart_output = gr.Plot(label="📈 Models Comparison")
            
            details_output = gr.Textbox(label="📋 Details", lines=15, interactive=False)
            
            predict_btn.click(
                fn=predict_patient,
                inputs=[patient_id_new, patient_name_new, age_new, sex_new, treatment_new,
                       crowding_up_new, crowding_low_new, overjet_new, anb_new],
                outputs=[card_output, chart_output, details_output]
            )
        
        # ============= TAB 2: UPDATE PATIENT =============
        with gr.Tab("✏️ Update Patient"):
            gr.Markdown("### Update patient with cooperation data")
            
            with gr.Row():
                patient_id_update = gr.Textbox(label="Patient ID", placeholder="P001")
                cooperation_update = gr.Radio(["Good", "Moderate", "Poor"], value="Good", label="Cooperation")
                missed_apt_update = gr.Slider(0, 20, value=0, label="Missed Appointments")
            
            update_btn = gr.Button("💾 Update & Generate Final Prediction", variant="primary", size="lg")
            
            with gr.Row():
                with gr.Column():
                    card_update = gr.Image(label="📊 Final Prediction Card")
                with gr.Column():
                    chart_update = gr.Plot(label="📈 Updated Comparison")
            
            details_update = gr.Textbox(label="📋 Update Details", lines=15, interactive=False)
            
            update_btn.click(
                fn=update_patient,
                inputs=[patient_id_update, cooperation_update, missed_apt_update],
                outputs=[card_update, chart_update, details_update]
            )
        
        # ============= TAB 3: HISTORY =============
        with gr.Tab("📈 Patient History"):
            gr.Markdown("### View all patients and predictions")
            
            history_btn = gr.Button("📂 Load History", variant="primary", size="lg")
            history_table = gr.Dataframe(label="Patient Records", interactive=False)
            
            history_btn.click(
                fn=get_history_table,
                outputs=history_table
            )
        
        # ============= TAB 4: ABOUT =============
        with gr.Tab("ℹ️ About"):
            gr.Markdown(f"""
            # 🦷 Orthodontic AI System
            
            ## Professional Prediction Engine
            AI-powered system for predicting orthodontic treatment duration
            
            ### 6 Machine Learning Models
            - Linear Regression
            - Lasso Regression
            - ElasticNet
            - Random Forest
            - AdaBoost
            - XGBoost (Best)
            
            ### Key Features
            ✅ Real-time predictions  
            ✅ Professional visual reports  
            ✅ Cooperation tracking  
            ✅ Patient database  
            ✅ Accuracy: ±{best_mae:.1f} months
            
            ### How to Use
            1. Enter patient information
            2. Input clinical measurements
            3. Get instant prediction
            4. Update with cooperation data for final estimate
            
            ---
            © 2025 Orthodontic AI - Developed for Clinical Excellence
            """)

if __name__ == "__main__":
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860
    )
