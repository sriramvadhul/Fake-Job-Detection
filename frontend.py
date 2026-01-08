import streamlit as st
import joblib
import re
import numpy as np
import random


st.set_page_config(
    page_title="Fake Job Posting Detector",
    page_icon="🕵️",
    layout="wide"
)

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    .title-badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(49, 51, 63, 0.2);
        background: rgba(49, 51, 63, 0.05);
        margin-bottom: 0.6rem;
    }
    .big-metric {
        font-size: 2.1rem;
        font-weight: 800;
        margin: 0.2rem 0 0.4rem 0;
    }
    .subtle {
        color: rgba(49, 51, 63, 0.75);
        font-size: 0.95rem;
        line-height: 1.3rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return joblib.load("best_fake_job_model.pkl")

model = load_model()


FIXED_THRESHOLD = 0.0

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"\S+@\S+", " ", text)          # emails
    text = re.sub(r"[^a-z\s]", " ", text)         # letters/spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def validate_input(raw_text: str) -> tuple[bool, str]:
    cleaned = clean_text(raw_text)
    if len(cleaned) < 30 or len(cleaned.split()) < 6:
        return False, "Please enter a meaningful job description (at least ~1–2 sentences)."
    return True, ""


# LinearSVC-aware predict (uses fixed threshold)

def predict(text: str):
    cleaned = clean_text(text)

    # Decision score from LinearSVC pipeline
    score = float(model.decision_function([cleaned])[0])

    # Pred based on fixed threshold (no user input)
    pred = 1 if score >= FIXED_THRESHOLD else 0

    # pseudo confidence for UI (sigmoid of score)
    fake_prob = 1 / (1 + np.exp(-score))
    real_prob = 1 - fake_prob

    return pred, real_prob, fake_prob, score

FAKE_JOBS = [
    "URGENT HIRING!!! Work from home and earn ₹50,000 per month. No experience required. No interview. Limited slots available. DM us immediately on WhatsApp to apply.",
    "Online assistant required. Simple typing work. Weekly payment guaranteed. Anyone can apply. Training provided. Apply now to start earning today.",
    "URGENT HIRING!!! Work from home and earn $500 to $1000 per day. No experience required. No interview. Contact us immediately on WhatsApp.",
    "Part-time remote job available. Flexible working hours. All you need is a laptop and internet connection. Earn extra income easily. Message us to apply.",
    "Immediate hiring for home-based work. No skills needed. No documents required. Earn money from your phone. Interested candidates DM fast."
]

REAL_JOBS = [
    "We are hiring a Data Analyst to join our analytics team. Responsibilities include data cleaning, dashboard creation, and reporting. Requirements include SQL, Python, and Power BI. Bachelor degree preferred. Competitive salary and benefits offered.",
    "HCL Technologies is seeking a Software Engineer. The role involves developing and maintaining backend services. Requirements include Java, Spring Boot, and REST APIs. Minimum two years of experience required. This is a full time position with health insurance and paid leave.",
    "Marketing Intern position available at XYZ Media. Responsibilities include social media content creation and campaign analysis. Candidates must be enrolled in a bachelor program. Internship duration is six months.",
    "We are looking for an Operations Coordinator to support daily business activities. Responsibilities include scheduling, reporting, and coordination with vendors. Strong communication and organizational skills required. Salary and benefits will be discussed during the interview.",
    "Customer Support Executive required for our service team. Responsibilities include handling customer queries via email and phone. Good communication skills and basic computer knowledge required. Full time position with fixed working hours."
]

BORDERLINE_JOBS = [
    "Driver required for night shifts on highways. Salary around ₹20,000 per month. Only experienced drivers preferred. Interested candidates may contact directly.",
    "Customer Service Representative needed. Work from home option available. Good communication skills required. Apply online.",
    "Office assistant required. Basic computer knowledge preferred. Training will be provided. Contact for more details.",
    "Sales executive needed for a local distribution business. Field work involved. Monthly salary plus incentives. Contact directly for interview details.",
    "Content reviewer required for an online platform. Remote work available. English reading and writing skills required. Payment based on number of tasks completed."
]

with st.sidebar:
    st.markdown("### 🧭 Project Overview")
    st.write(
        "This demo classifies a job posting as **Real** or **Fake** using "
        "**NLP (TF-IDF)** + **LinearSVC**."
    )

    st.markdown("### 🧪 Demo Inputs")
    if st.button("Load FAKE example"):
        st.session_state["job_text"] = FAKE_JOBS[random.randint(0, 4)]

    if st.button("Load REAL example"):
        st.session_state["job_text"] = REAL_JOBS[random.randint(0, 4)]

    if st.button("Load BORDERLINE example"):
        st.session_state["job_text"] = BORDERLINE_JOBS[random.randint(0, 4)]
    


st.markdown('<div class="title-badge">NLP • Binary Classification • Supervised ML</div>', unsafe_allow_html=True)
st.title("🕵️ Fake Job Posting Detector")
st.write(
    "Paste a job posting and click **Analyze**. The app returns a prediction and confidence signal. "
)


left, right = st.columns([1.25, 1])

with left:
    st.subheader("📄 Input: Job Posting Text")
    job_text = st.text_area(
        label="",
        height=320,
        placeholder="Paste the job description here...",
        value=st.session_state.get("job_text", "")
    )

    col_a, col_b = st.columns([1, 1])
    with col_a:
        analyze = st.button("🔍 Analyze", use_container_width=True)
    with col_b:
        clear = st.button("🧹 Clear", use_container_width=True)

    if clear:
        st.session_state["job_text"] = ""
        st.rerun()

with right:
    st.subheader(" Output: Prediction")
    st.write("")

    if analyze:
        if not job_text.strip():
            st.warning("Please paste a job posting first.")
        else:
            is_valid, msg = validate_input(job_text)
            if not is_valid:
                st.warning(msg)
                st.stop()

            pred, real_prob, fake_prob, score = predict(job_text)

            if pred == 1:
                st.error("**FAKE JOB POSTING 🚨**")
            else:
                st.success("**REAL JOB POSTING ✅**")

            st.markdown("#### 📊 Confidence Signal")
            st.write(f"SVC decision score: **{score:.3f}**  |  fixed threshold: **{FIXED_THRESHOLD:.2f}**")

            st.progress(fake_prob if pred == 1 else real_prob)
            c1, c2 = st.columns(2)
            c1.metric("Real", f"{real_prob:.1%}")
            c2.metric("Fake", f"{fake_prob:.1%}")

