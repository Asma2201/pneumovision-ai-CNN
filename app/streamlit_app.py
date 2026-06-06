import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
from PIL import Image
from predict import load_model, predict
from model import get_device
from fpdf import FPDF
import datetime


# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────

st.set_page_config(
    page_title = "PneumoVision AI",
    page_icon  = "🫁",
    layout     = "wide"
)

st.markdown("""
<style>
    /* ── Global ── */
    html, body, [class*="css"] {
        font-size: 16px !important;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a2540 0%, #0f3460 60%, #1a4a7a 100%);
        padding: 0;
    }
    [data-testid="stSidebar"] * { color: white !important; }
    .main { background-color: #f0f4f8; }
    .block-container { padding: 2rem 2.5rem; }

    /* ── Sidebar branding ── */
    .sidebar-brand {
        background: rgba(255,255,255,0.08);
        border-bottom: 1px solid rgba(255,255,255,0.12);
        padding: 24px 20px 20px;
        margin-bottom: 12px;
        text-align: center;
    }
    .sidebar-brand h1 {
        font-size: 22px !important;
        font-weight: 700;
        margin: 8px 0 4px;
        color: white !important;
    }
    .sidebar-brand p {
        font-size: 12px !important;
        color: rgba(255,255,255,0.6) !important;
        margin: 0;
    }
    .sidebar-badge {
        display: inline-block;
        background: rgba(34,197,94,0.2);
        border: 1px solid rgba(34,197,94,0.4);
        color: #86efac !important;
        font-size: 11px !important;
        padding: 3px 10px;
        border-radius: 20px;
        margin-top: 8px;
    }
    .sidebar-stat {
        background: rgba(255,255,255,0.07);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 16px;
        font-size: 13px !important;
    }
    .sidebar-stat span {
        font-size: 18px !important;
        font-weight: 700;
        display: block;
        color: #7dd3fc !important;
    }

    /* ── Page header ── */
    .page-header {
        background: white;
        border-radius: 16px;
        padding: 24px 28px;
        margin-bottom: 24px;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .page-header h1 {
        font-size: 26px !important;
        font-weight: 700;
        color: #0f172a;
        margin: 0 0 4px;
    }
    .page-header p {
        font-size: 14px !important;
        color: #64748b;
        margin: 0;
    }

    /* ── Section labels ── */
    .section-label {
        font-size: 11px !important;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .section-num {
        background: #0f3460;
        color: white !important;
        font-size: 11px !important;
        width: 20px; height: 20px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
    }

    /* ── Cards ── */
    .card {
        background: white;
        border-radius: 14px;
        padding: 22px 24px;
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .card h3 {
        font-size: 13px !important;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin: 0 0 14px;
    }

    /* ── Result styles ── */
    .result-pneumonia {
        font-size: 28px !important;
        font-weight: 800;
        color: #dc2626;
        margin: 0 0 4px;
    }
    .result-normal {
        font-size: 28px !important;
        font-weight: 800;
        color: #16a34a;
        margin: 0 0 4px;
    }
    .confidence-label {
        font-size: 13px !important;
        color: #64748b;
        margin-bottom: 16px;
    }
    .big-score {
        font-size: 42px !important;
        font-weight: 800;
        color: #0f172a;
        line-height: 1;
    }
    .score-label {
        font-size: 13px !important;
        color: #94a3b8;
        margin-top: 4px;
    }

    /* ── Status badge ── */
    .status-high {
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .status-high h2 {
        font-size: 20px !important;
        font-weight: 700;
        color: #dc2626;
        margin: 0 0 4px;
    }
    .status-high p {
        font-size: 13px !important;
        color: #ef4444;
        margin: 0;
    }
    .status-low {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 10px;
        padding: 14px 16px;
        margin-bottom: 14px;
    }
    .status-low h2 {
        font-size: 20px !important;
        font-weight: 700;
        color: #16a34a;
        margin: 0 0 4px;
    }
    .status-low p {
        font-size: 13px !important;
        color: #22c55e;
        margin: 0;
    }
    .finding-item {
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 10px 0;
        border-bottom: 1px solid #f1f5f9;
        font-size: 14px !important;
        color: #374151;
    }
    .finding-dot-red  { width:8px; height:8px; background:#dc2626; border-radius:50%; margin-top:5px; flex-shrink:0; }
    .finding-dot-green{ width:8px; height:8px; background:#16a34a; border-radius:50%; margin-top:5px; flex-shrink:0; }

    /* ── Action buttons ── */
    .stButton > button {
        font-size: 15px !important;
        font-weight: 600 !important;
        padding: 10px 0 !important;
        border-radius: 10px !important;
    }

    /* ── Patient form ── */
    .stTextInput label, .stNumberInput label, .stRadio label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #374151 !important;
    }
    .stTextInput input, .stNumberInput input {
        font-size: 15px !important;
        padding: 10px 14px !important;
        border-radius: 8px !important;
    }

    /* ── Footer ── */
    .footer {
        background: #1e293b;
        border-radius: 10px;
        padding: 14px 20px;
        margin-top: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .footer p {
        font-size: 12px !important;
        color: #94a3b8;
        margin: 0;
    }
    .footer span {
        font-size: 12px !important;
        color: #7dd3fc;
        font-weight: 600;
    }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] {
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 12px !important;
        background: #f8fafc !important;
    }
    [data-testid="stFileUploader"] * {
        font-size: 14px !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# MODELE
# ─────────────────────────────────────────

@st.cache_resource
def get_model():
    device = get_device()
    model  = load_model("models/model.pth", device)
    return model, device


# ─────────────────────────────────────────
# PDF
# ─────────────────────────────────────────

def generate_pdf(patient_id, age, gender, label, score):
    pdf = FPDF()
    pdf.add_page()

    # Header
    pdf.set_fill_color(15, 52, 96)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font("Arial", "B", 20)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(0, 15, "", ln=True)
    pdf.cell(0, 12, "  PneumoVision AI - Rapport Medical", ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Infos patient
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(240, 244, 248)
    pdf.cell(0, 8, "Informations Patient", ln=True, fill=True)
    pdf.ln(4)
    pdf.set_font("Arial", size=12)
    pdf.cell(0, 8, f"  Patient ID   : {patient_id}", ln=True)
    pdf.cell(0, 8, f"  Age          : {age} ans", ln=True)
    pdf.cell(0, 8, f"  Genre        : {gender}", ln=True)
    pdf.cell(0, 8, f"  Date         : {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}", ln=True)
    pdf.ln(8)

    # Résultat
    pdf.set_font("Arial", "B", 13)
    pdf.set_fill_color(240, 244, 248)
    pdf.cell(0, 8, "Resultat de l'Analyse", ln=True, fill=True)
    pdf.ln(4)
    pdf.set_font("Arial", "B", 16)
    if label == "PNEUMONIA":
        pdf.set_text_color(220, 38, 38)
        pdf.cell(0, 10, f"  Diagnostic   : PNEUMONIE DETECTEE", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"  Confiance    : {score:.2f}%", ln=True)
        pdf.cell(0, 8,  "  Statut       : Risque Eleve", ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", "I", 11)
        pdf.set_text_color(220, 38, 38)
        pdf.cell(0, 8, "  Consultation medicale immediate recommandee.", ln=True)
    else:
        pdf.set_text_color(22, 163, 74)
        pdf.cell(0, 10, f"  Diagnostic   : NORMAL", ln=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 8, f"  Confiance    : {score:.2f}%", ln=True)
        pdf.cell(0, 8,  "  Statut       : Risque Faible", ln=True)
        pdf.ln(6)
        pdf.set_font("Arial", "I", 11)
        pdf.set_text_color(22, 163, 74)
        pdf.cell(0, 8, "  Aucune anomalie detectee. Suivi de routine recommande.", ln=True)

    # Footer PDF
    pdf.set_text_color(0, 0, 0)
    pdf.ln(16)
    pdf.set_font("Arial", "I", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(0, 6, "Rapport genere par PneumoVision AI | ResNet18 Transfer Learning | Acc: 95.11%", ln=True, align="C")

    path = f"rapport_{patient_id}.pdf"
    pdf.output(path)
    return path


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div style="font-size:42px">🫁</div>
        <h1>PneumoVision AI</h1>
        <p>Medical Imaging Assistant</p>
        <div class="sidebar-badge">● System Online</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "",
        ["🔬  Scan Analysis", "📋  Patient History", "📊  Reports", "ℹ️  About AI"],
        label_visibility="collapsed"
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-stat">
        Accuracy<span>95.11%</span>
    </div>
    <div class="sidebar-stat">
        Architecture<span>ResNet-18</span>
    </div>
    <div class="sidebar-stat">
        Framework<span>PyTorch</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# PAGE SCAN ANALYSIS
# ─────────────────────────────────────────

if "Scan Analysis" in page:

    st.markdown("""
    <div class="page-header">
        <div style="font-size:48px">🩻</div>
        <div>
            <h1>Pneumonia Detection Dashboard</h1>
            <p>Upload a chest X-ray scan and enter patient details to get an AI-powered diagnosis.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section 1 + 2 ──
    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown("""
        <div class="section-label">
            <span class="section-num">1</span> Upload Patient Scan
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag and drop or browse a chest X-ray",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="📷 Radiographie chargée", use_column_width=True)

    with col2:
        st.markdown("""
        <div class="section-label">
            <span class="section-num">2</span> Patient Details
        </div>
        """, unsafe_allow_html=True)

        patient_id = st.text_input("🪪 Patient ID", value="P00001")
        col_a, col_b = st.columns(2)
        with col_a:
            age = st.number_input("🎂 Age", min_value=1, max_value=120, value=45)
        with col_b:
            gender = st.radio("⚧ Gender", ["Male", "Female"], horizontal=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px;">
            <p style="font-size:12px; color:#94a3b8; margin:0 0 8px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Patient Summary</p>
            <p style="font-size:15px; color:#0f172a; margin:0; font-weight:600;">ID: {patient_id}</p>
            <p style="font-size:14px; color:#64748b; margin:4px 0 0;">{age} years old · {gender}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section-label">
        <span class="section-num">3</span> Prediction Results
    </div>
    """, unsafe_allow_html=True)

    # ── Section 3 : Résultats ──
    if uploaded_file:
        temp_path = "temp_image.jpg"
        image.save(temp_path)

        with st.spinner("🔍 Analyzing scan..."):
            model, device = get_model()
            label, score  = predict(temp_path, model, device)

        os.remove(temp_path)

        col_r, col_s, col_a = st.columns(3)

        # Carte résultat
        with col_r:
            if label == "PNEUMONIA":
                st.markdown(f"""
                <div class="card">
                    <h3>🔬 Prediction</h3>
                    <p class="result-pneumonia">⚠️ Pneumonia</p>
                    <p class="confidence-label">High confidence detection</p>
                    <p class="big-score">{score:.1f}%</p>
                    <p class="score-label">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card">
                    <h3>🔬 Prediction</h3>
                    <p class="result-normal">✅ Normal</p>
                    <p class="confidence-label">No anomaly detected</p>
                    <p class="big-score">{score:.1f}%</p>
                    <p class="score-label">Confidence Score</p>
                </div>
                """, unsafe_allow_html=True)

        # Carte statut
        with col_s:
            if label == "PNEUMONIA":
                st.markdown("""
                <div class="card">
                    <h3>🏥 Patient Status</h3>
                    <div class="status-high">
                        <h2>🔴 High Risk</h2>
                        <p>Immediate attention required</p>
                    </div>
                    <div class="finding-item">
                        <div class="finding-dot-red"></div>
                        <span>Potential infiltration detected</span>
                    </div>
                    <div class="finding-item">
                        <div class="finding-dot-red"></div>
                        <span>Pulmonary opacity observed</span>
                    </div>
                    <div class="finding-item">
                        <div class="finding-dot-red"></div>
                        <span>Specialist consultation advised</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="card">
                    <h3>🏥 Patient Status</h3>
                    <div class="status-low">
                        <h2>🟢 Low Risk</h2>
                        <p>No immediate action needed</p>
                    </div>
                    <div class="finding-item">
                        <div class="finding-dot-green"></div>
                        <span>No infiltration detected</span>
                    </div>
                    <div class="finding-item">
                        <div class="finding-dot-green"></div>
                        <span>Lung fields appear clear</span>
                    </div>
                    <div class="finding-item">
                        <div class="finding-dot-green"></div>
                        <span>Routine follow-up recommended</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Carte actions
        with col_a:
            st.markdown('<div class="card"><h3>⚡ AI Recommendation</h3>', unsafe_allow_html=True)

            if label == "PNEUMONIA":
                st.error("🚨 Immediate Medical Review")
            else:
                st.success("✅ No Immediate Action")

            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("📄 Generate Report (PDF)", use_container_width=True, type="primary"):
                pdf_path = generate_pdf(patient_id, age, gender, label, score)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label               = "⬇️ Download PDF Report",
                        data                = f,
                        file_name           = f"PneumoVision_{patient_id}.pdf",
                        mime                = "application/pdf",
                        use_container_width = True
                    )
                os.remove(pdf_path)

            st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown(f"""
        <div class="footer">
            <p>Analysis Complete · ResNet18 Transfer Learning · <span>Acc: 95.11%</span></p>
            <p>Patient: <span>{patient_id}</span> · {datetime.datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="background:white; border:2px dashed #cbd5e1; border-radius:14px;
                    padding:48px; text-align:center; color:#94a3b8;">
            <div style="font-size:56px; margin-bottom:16px;">🩻</div>
            <p style="font-size:18px; font-weight:600; color:#64748b; margin:0 0 8px;">
                No scan uploaded yet
            </p>
            <p style="font-size:14px; margin:0;">
                Upload a chest X-ray in section 1 to start the AI analysis
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────
# AUTRES PAGES
# ─────────────────────────────────────────

elif "About AI" in page:
    st.markdown("""
    <div class="page-header">
        <div style="font-size:48px">ℹ️</div>
        <div>
            <h1>About PneumoVision AI</h1>
            <p>Technical details about the model and dataset.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="card">
            <h3>🧠 Model Details</h3>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Architecture</b> : ResNet-18</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Method</b> : Transfer Learning</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Framework</b> : PyTorch</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Val Accuracy</b> : 95.11%</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Classes</b> : NORMAL / PNEUMONIA</span></div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="card">
            <h3>📦 Dataset</h3>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Source</b> : Kaggle Chest X-Ray</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Train images</b> : 5,216</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Test images</b> : 624</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Input size</b> : 224×224 px</span></div>
            <div class="finding-item"><div class="finding-dot-green"></div><span><b>Normalization</b> : ImageNet</span></div>
        </div>
        """, unsafe_allow_html=True)

elif "Reports" in page:
    st.markdown("""
    <div class="page-header">
        <div style="font-size:48px">📊</div>
        <div><h1>Reports</h1><p>Generated PDF reports will appear here.</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.info("No reports generated yet. Run a scan analysis first.")

elif "Patient History" in page:
    st.markdown("""
    <div class="page-header">
        <div style="font-size:48px">📋</div>
        <div><h1>Patient History</h1><p>Previous scan results will appear here.</p></div>
    </div>
    """, unsafe_allow_html=True)
    st.info("No patient history yet. Run a scan analysis first.")