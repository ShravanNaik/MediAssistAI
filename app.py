


import sys
import importlib
importlib.import_module('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


import streamlit as st
from crewai import Agent, Task, Crew, LLM, Process
import os
from crewai_tools import ScrapeWebsiteTool, SerperDevTool, FileReadTool, DirectoryReadTool
from dotenv import load_dotenv
import base64
from PIL import Image
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from datetime import datetime, timedelta
from langchain_openai import ChatOpenAI
from pathlib import Path


# Load environment variables
load_dotenv()
os.environ["SERPER_API_KEY"] = os.getenv("SERPER_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Create output directory if it doesn't exist
output_dir = Path("task_outputs")
output_dir.mkdir(exist_ok=True)
temp_output_dir = "temp_outputs"
os.makedirs(temp_output_dir, exist_ok=True)



# Function to load and display logo
def load_logo():
    return "https://img.icons8.com/color/96/000000/caduceus.png"
# Function for page styling
def apply_custom_styling():
    st.markdown("""
    <style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #2c3e50;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 0.5em;
    }
    .sub-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #3498db;
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 1.5em;
    }
    .diagnosis-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1em;
    }
    .treatment-box {
        background-color: #e8f4f8;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1em;
    }
    .research-box {
        background-color: #f0f7ee;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1em;
    }
    .alert-box {
        background-color: #fde9e8;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #e74c3c;
        margin-bottom: 1em;
    }
    .info-box {
        background-color: #e8f4fd;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #3498db;
        margin-bottom: 1em;
    }
    .note-box {
        background-color: #fef9e7;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 3px solid #f1c40f;
        font-size: 0.9em;
        margin-bottom: 1em;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-weight: bold;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #2980b9;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f0f2f6;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        height: 60px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db !important;
        color: white !important;
    }
    div[data-testid="stExpander"] div[role="button"] p {
        font-size: 1.05em;
        font-weight: 500;
    }
    .sidebar-content {
        padding: 15px;
    }
    .highlight-text {
        color: #3498db;
        font-weight: 600;
    }
    .card {
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: white;
        transition: transform 0.3s ease;
        color: #333333;  /* Dark text color for better readability */
        min-height: 100px;
        border: 1px solid #e0e0e0;
    }
    .card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .card h3 {
        color: #2c3e50;
        margin-top: 0;
        padding-bottom: 10px;
        border-bottom: 1px solid #eee;
    }

    .card h4 {
        color: #3498db;
        margin-top: 0;
    }
    .card p {
        margin-bottom: 0;
    }
    .markdown-text-container {
        color: #333333 !important;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(0, 0, 0, 0.1);
    }
    .stMarkdown {
        color: #333333 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to display a progress animation
def progress_animation():
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    stages = [
        "Initializing medical analysis system...",
        "Primary diagnosis agent analyzing symptoms...",
        "Consulting medical knowledge databases...",
        "Specialist agents reviewing findings...",
        "Treatment advisor generating recommendations...",
        "Medical researcher identifying relevant studies...",
        "Safety verification in progress...",
        "Pharmacology agent reviewing medication interactions...",
        "Patient education specialist preparing materials...",
        "Finalizing comprehensive medical report..."
    ]
    
    for i, stage in enumerate(stages):
        status_text.text(stage)
        progress_bar.progress((i+1)/len(stages))
        time.sleep(0.7)
    
    progress_bar.empty()
    status_text.empty()

# Function to save task output to markdown file
def save_task_output(task_name, output):
    filename = f"{output_dir}/{task_name}.md"
    output_str = str(output)  
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(output_str)
    
    # Ensure alerts are properly captured if they exist in the output
    if "## ALERTS" in output_str:
        st.session_state.task_output_files["alerts"] = filename
    return filename

# Function to parse markdown content
def parse_markdown_content(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Initialize sections
        sections = {
            "diagnosis": "",
            "treatment": "",
            "research": "",
            "alerts": "",
            "patient_education": ""
        }
        
        # Extract sections
        if "## DIAGNOSIS" in content:
            diagnosis_section = content.split("## DIAGNOSIS")[1].split("## TREATMENT PLAN")[0]
            sections["diagnosis"] = diagnosis_section.strip()
        
        if "## TREATMENT PLAN" in content:
            if "## MEDICAL RESEARCH" in content:
                treatment_section = content.split("## TREATMENT PLAN")[1].split("## MEDICAL RESEARCH")[0]
            else:
                treatment_section = content.split("## TREATMENT PLAN")[1]
            sections["treatment"] = treatment_section.strip()
        
        if "## MEDICAL RESEARCH" in content:
            if "## ALERTS" in content:
                research_section = content.split("## MEDICAL RESEARCH")[1].split("## ALERTS")[0]
            else:
                research_section = content.split("## MEDICAL RESEARCH")[1]
            sections["research"] = research_section.strip()
        
        if "## ALERTS" in content:
            if "## PATIENT EDUCATION" in content:
                alerts_section = content.split("## ALERTS")[1].split("## PATIENT EDUCATION")[0]
            else:
                alerts_section = content.split("## ALERTS")[1]
            sections["alerts"] = alerts_section.strip()
        
        if "## PATIENT EDUCATION" in content:
            education_section = content.split("## PATIENT EDUCATION")[1]
            sections["patient_education"] = education_section.strip()
            
        return sections
        
    except Exception as e:
        st.error(f"Error parsing markdown content: {str(e)}")
        return {
            "diagnosis": "Error parsing diagnosis results.",
            "treatment": "Error parsing treatment results.",
            "research": "Error parsing research results.",
            "alerts": "Error parsing alerts.",
            "patient_education": "Error parsing patient education materials."
        }

# Function to generate prescription document
def generate_prescription(patient_name, gender, age, diagnosis, medications, doctor_name="AI Doctor Assistant"):
    prescription = f"""
    <div style="width: 800px; padding: 20px; border: 2px solid #3498db; border-radius: 10px; font-family: Arial, sans-serif;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 20px;">
            <div>
                <h2 style="color: #3498db; margin: 0;">MediAssist AI Clinic</h2>
                <p style="margin: 5px 0;">123 Healthcare Avenue</p>
                <p style="margin: 5px 0;">Medical District, MD 12345</p>
                <p style="margin: 5px 0;">Phone: (555) 123-4567</p>
            </div>
            <div>
                <h2 style="color: #3498db; text-align: right;">PRESCRIPTION</h2>
                <p style="text-align: right; margin: 5px 0;">Date: {datetime.now().strftime('%B %d, %Y')}</p>
                <p style="text-align: right; margin: 5px 0;">Rx #: {int(time.time())}</p>
            </div>
        </div>
        
        <div style="padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px 0; color: #2c3e50;">Patient Information</h3>
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 5px; width: 25%;"><strong>Name:</strong></td>
                    <td style="padding: 5px;">{patient_name}</td>
                    <td style="padding: 5px; width: 25%;"><strong>Gender:</strong></td>
                    <td style="padding: 5px;">{gender}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Age:</strong></td>
                    <td style="padding: 5px;">{age} years</td>
                    <td style="padding: 5px;"><strong>Date:</strong></td>
                    <td style="padding: 5px;">{datetime.now().strftime('%m/%d/%Y')}</td>
                </tr>
                <tr>
                    <td style="padding: 5px;"><strong>Diagnosis:</strong></td>
                    <td style="padding: 5px;" colspan="3">{diagnosis}</td>
                </tr>
            </table>
        </div>
        
        <div style="margin-bottom: 20px;">
            <h3 style="margin: 0 0 10px 0; color: #2c3e50;">Rx</h3>
            <div style="border-left: 3px solid #3498db; padding-left: 15px;">
    """
    
    # Add medications
    if isinstance(medications, list):
        for med in medications:
            prescription += f"""
                <p style="margin: 10px 0;">{med}</p>
                <hr style="border-top: 1px dashed #ddd; margin: 10px 0;">
            """
    else:
        prescription += f"""
            <p style="margin: 10px 0;">{medications}</p>
        """
    
    prescription += f"""
            </div>
        </div>
        
        <div style="display: flex; justify-content: space-between; margin-top: 40px;">
            <div>
                <p style="border-top: 1px solid #2c3e50; padding-top: 5px; width: 200px;">Physician Signature</p>
                <p><strong>Dr. {doctor_name}</strong></p>
                <p>License #: AI-MD-12345</p>
            </div>
            <div>
                <p style="text-align: right; font-style: italic; color: #7f8c8d;">This prescription was generated with AI assistance.</p>
                <p style="text-align: right; font-style: italic; color: #7f8c8d;">Please consult with a licensed healthcare provider.</p>
            </div>
        </div>
    </div>
    """
    
    return prescription

# Extract medications from treatment plan
def extract_medications(treatment_text):
    medications = []
    
    # Look for medication patterns
    med_patterns = [
        r"(\d+\.\s*[A-Za-z]+\s+\d+\s*mg\s*[a-zA-Z0-9\s,]+)",
        r"([A-Za-z]+\s+\d+\s*mg\s*[a-zA-Z0-9\s,]+)",
        r"(Prescribe\s+[A-Za-z]+\s+\d+\s*mg\s*[a-zA-Z0-9\s,]+)"
    ]
    
    for pattern in med_patterns:
        matches = re.findall(pattern, treatment_text)
        if matches:
            medications.extend(matches)
    
    # If no structured medications found, look for bullet points or numbered lists
    if not medications:
        lines = treatment_text.split("\n")
        for line in lines:
            if ("mg" in line or "tablet" in line or "capsule" in line) and ("take" in line.lower() or "daily" in line.lower() or "twice" in line.lower()):
                medications.append(line.strip())
    
    # If still no medications found, return placeholder
    if not medications:
        return ["Medications to be determined by physician based on final diagnosis."]
    
    return medications

# Initialize session state variables if they don't exist
if 'past_consultations' not in st.session_state:
    st.session_state.past_consultations = []
if 'task_output_files' not in st.session_state:
    st.session_state.task_output_files = {}

# Streamlit UI Configuration
st.set_page_config(
    page_title="MediAssist AI", 
    layout="wide", 
    page_icon="🩺"
)

hide_footer_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* This targets GitHub icon in the footer */
    .st-emotion-cache-1y4p8pa.ea3mdgi1 {
        display: none !important;
    }

    /* This targets the entire footer area */
    .st-emotion-cache-164nlkn {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_footer_style, unsafe_allow_html=True)

hide_streamlit_style = """
    <style>
    /* Hide hamburger menu and header */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}

    /* Hide entire footer */
    footer {visibility: hidden;}
    .st-emotion-cache-zq5wmm {display: none;}
    .st-emotion-cache-13ln4jf {display: none;}

    /* Extra fallback for "Made with Streamlit" footer */
    .viewerBadge_container__1QSob {
        display: none !important;
    }
    .stDeployButton {
        display: none !important;
    }
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)



apply_custom_styling()

# Sidebar Navigation
with st.sidebar:
    st.image(load_logo(), width=100)
    st.title("MediAssist AI")
    st.markdown("**Advanced Healthcare Intelligence Platform**")
    st.divider()
    
    # Navigation
    page = st.radio("Navigation", ["Home", "New Consultation", "Past Consultations", "Medical Knowledge", "About"])
    
    st.divider()
    st.markdown("## Features")
    st.markdown("✅ Multi-Agent Medical Intelligence")
    st.markdown("✅ Comprehensive Diagnostic Support")
    st.markdown("✅ Evidence-Based Treatment Plans")
    st.markdown("✅ Medical Literature Integration")
    st.markdown("✅ Patient Education Materials")
    st.markdown("✅ Medication Safety Analysis")
    
    st.divider()
    st.markdown("*Disclaimer: This tool is for informational purposes only and does not replace professional medical advice.*")

# Initialize Tools
search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()
file_tool = FileReadTool()
directory_tool = DirectoryReadTool()

# Initialize LLM
# llm = LLM(model="openai/gpt-4o-mini", temperature=0.7, api_key=os.environ["OPENAI_API_KEY"])
llm = ChatOpenAI(model = "gpt-4o-mini")


# Define Agents with more detailed roles, goals, and delegation capabilities
primary_diagnostician = Agent(
    role="Primary Medical Diagnostician",
    goal="Coordinate the diagnostic process and generate accurate preliminary diagnoses based on patient information.",
    backstory="""As the Primary Medical Diagnostician, you have extensive clinical experience and 
    are responsible for coordinating the initial diagnostic process. You analyze patient symptoms, 
    medical history, and vital signs to develop a comprehensive differential diagnosis. You're skilled at 
    identifying patterns in symptoms and prioritizing potential conditions based on likelihood.""",
    llm=llm,
    verbose=True,
    max_iter=2,
    allow_delegation=True,
    tools=[search_tool, scrape_tool]
)

specialist_diagnostician = Agent(
    role="Specialist Diagnostician",
    goal="Provide specialized expertise for complex or specific medical conditions based on the primary diagnostician's findings.",
    backstory="""You are a medical specialist with deep expertise in complex conditions. When the primary 
    diagnostician identifies potential specialized conditions, you provide in-depth analysis and expertise. 
    Your specialized knowledge allows for more accurate diagnosis of complex or rare conditions.""",
    llm=llm,
    verbose=True,
    max_iter=2,
    allow_delegation=True,
    tools=[search_tool, scrape_tool]
)

treatment_advisor = Agent(
    role="Treatment Planning Specialist",
    goal="Develop comprehensive, personalized treatment plans based on confirmed diagnoses and patient profiles.",
    backstory="""As a Treatment Planning Specialist, you excel at creating individualized treatment strategies. 
    You consider the diagnosis, patient profile, current medications, allergies, and best medical practices 
    to develop appropriate treatment plans. Your recommendations are evidence-based and tailored to the 
    specific needs of each patient.""",
    llm=llm,
    verbose=True,
    max_iter=2,
    allow_delegation=True,
    tools=[search_tool, scrape_tool]
)

pharmacology_specialist = Agent(
    role="Clinical Pharmacologist",
    goal="Ensure medication safety by analyzing potential drug interactions and providing dosage guidance.",
    backstory="""You are a clinical pharmacology expert who specializes in medication safety and efficacy. 
    You analyze potential drug interactions, recommend appropriate dosages based on patient factors, 
    and ensure that medication plans are both safe and effective. You can identify contraindications 
    and suggest alternatives when necessary.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2,
    tools=[search_tool, scrape_tool]
)

medical_researcher = Agent(
    role="Medical Literature Researcher",
    goal="Find and synthesize relevant medical research to support diagnosis and treatment recommendations.",
    backstory="""You excel at searching medical databases and recent publications to find 
    evidence-based information relevant to specific patient cases. You evaluate the quality 
    of research and extract actionable insights to support medical decision-making. You ensure 
    that all recommendations are backed by current medical literature.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2,
    tools=[search_tool, scrape_tool]
)

patient_educator = Agent(
    role="Patient Education Specialist",
    goal="Create clear, accessible education materials for patients about their condition and treatment.",
    backstory="""You specialize in translating complex medical information into clear, 
    actionable guidance for patients. You create educational materials that help patients 
    understand their condition, treatment plan, and necessary lifestyle modifications. 
    Your communication is empathetic, clear, and designed to improve treatment adherence.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2,
    tools=[search_tool, scrape_tool]
)

safety_officer = Agent(
    role="Medical Safety Officer",
    goal="Identify potential risks in diagnosis and treatment and provide safety alerts.",
    backstory="""You are focused on patient safety in all aspects of care. You review diagnoses 
    and treatment plans to identify potential risks, warning signs that require immediate attention, 
    and safety precautions. Your role is to ensure that all recommendations prioritize patient safety 
    and include appropriate monitoring and follow-up.""",
    llm=llm,
    verbose=True,
    allow_delegation=False,
    max_iter=2,
    tools=[search_tool, scrape_tool]
)

# Content for different pages
if page == "Home":
    # Hero Section using Streamlit components
    st.markdown("""
    <style>
    .hero-container {
        background: linear-gradient(135deg, #3498db, #2ecc71);
        padding: 2.5rem;
        border-radius: 15px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    </style>
    <div class="hero-container">
        <h1 style="color: white; margin-bottom: 0.5rem;">MediAssist AI</h1>
        <h3 style="color: white; margin-top: 0; font-weight: 300;">
        Advanced Multi-Agent Medical Intelligence Platform
        </h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Add the powered by tag with Streamlit
    st.markdown(
        '<div style="text-align: center; margin-top: -30px; margin-bottom: 30px;">'
        '<span style="background-color: rgba(255,255,255,0.2); padding: 0.5rem 1.5rem; border-radius: 30px; font-weight: 500;">'
        'Powered by AI Medical Agents'
        '</span>'
        '</div>', 
        unsafe_allow_html=True
    )
    
    # Main Content
  # Main Content - Centered
    with st.container():
        # Create a centered container using CSS
        st.markdown("""
        <style>
        .centered-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 0 20px;
        }
        </style>
        <div class="centered-container">
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Welcome Section using Streamlit components
            with st.container():
                st.markdown("### Welcome to the Future of Medical Decision Support")
                st.markdown("""
                MediAssist AI leverages a sophisticated multi-agent system powered by artificial intelligence 
                to provide comprehensive support for healthcare professionals.
                """)
                
                # Info box using Streamlit
                st.info("""
                ⚡ Quick, accurate, and evidence-based medical insights
                """)
            
            # Features Section using Streamlit components
            st.markdown("### Key Features")
            features_col1, features_col2 = st.columns(2)
            
            with features_col1:
                # Feature 1
                with st.container():
                    st.markdown("#### 🔍 Advanced Diagnostics")
                    st.markdown("Multi-agent collaboration for thorough diagnostic assessments")
                
                # Feature 2
                with st.container():
                    st.markdown("#### ⚠️ Safety Alerts")
                    st.markdown("Critical warnings about drug interactions and contraindications")
            
            with features_col2:
                # Feature 3
                with st.container():
                    st.markdown("#### 💊 Treatment Plans")
                    st.markdown("Personalized treatment strategies based on latest guidelines")
                
                # Feature 4
                with st.container():
                    st.markdown("#### 📚 Medical Research")
                    st.markdown("Automatic integration of relevant medical literature")
            
            # How It Works Section using Streamlit components
            st.markdown("### How It Works")
            
            # Step 1
            col1_step, col2_step = st.columns([0.1, 0.9])
            with col1_step:
                st.markdown("""
                <div style="
                    background-color: #3498db;
                    color: white;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 1rem;
                ">1</div>
                """, unsafe_allow_html=True)
            with col2_step:
                st.markdown("#### Enter Patient Information")
                st.markdown("Provide symptoms, medical history, and other relevant details")
            
            # Step 2
            col1_step, col2_step = st.columns([0.1, 0.9])
            with col1_step:
                st.markdown("""
                <div style="
                    background-color: #3498db;
                    color: white;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 1rem;
                ">2</div>
                """, unsafe_allow_html=True)
            with col2_step:
                st.markdown("#### AI Agents Analyze")
                st.markdown("Specialized AI agents collaborate to assess the case")
            
            # Step 3
            col1_step, col2_step = st.columns([0.1, 0.9])
            with col1_step:
                st.markdown("""
                <div style="
                    background-color: #3498db;
                    color: white;
                    width: 30px;
                    height: 30px;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    margin-right: 1rem;
                ">3</div>
                """, unsafe_allow_html=True)
            with col2_step:
                st.markdown("#### Receive Comprehensive Report")
                st.markdown("Get diagnosis, treatment plan, research, and safety alerts")
        
        # with col2:
        #     # Quick Start Card using Streamlit
        #     if st.button("Start New Consultation", key="new_consult_btn", use_container_width=True):
        #         st.session_state['page'] = "New Consultation"
        #         st.rerun()
            
        #     # Stats Card using Streamlit
        #     st.markdown("### Platform Stats")
            
        #     # Stat 1
        #     col1_stat, col2_stat = st.columns([0.2, 0.8])
        #     with col1_stat:
        #         st.markdown("""
        #         <div style="
        #             background-color: rgba(52,152,219,0.1);
        #             width: 50px;
        #             height: 50px;
        #             border-radius: 50%;
        #             display: flex;
        #             align-items: center;
        #             justify-content: center;
        #         ">
        #             <span style="font-size: 1.5rem; color: #3498db;">⚕️</span>
        #         </div>
        #         """, unsafe_allow_html=True)
        #     with col2_stat:
        #         st.markdown("**7**")
        #         st.markdown("Specialized AI Agents")
            
        #     # Stat 2
        #     col1_stat, col2_stat = st.columns([0.2, 0.8])
        #     with col1_stat:
        #         st.markdown("""
        #         <div style="
        #             background-color: rgba(46,204,113,0.1);
        #             width: 50px;
        #             height: 50px;
        #             border-radius: 50%;
        #             display: flex;
        #             align-items: center;
        #             justify-content: center;
        #         ">
        #             <span style="font-size: 1.5rem; color: #2ecc71;">📊</span>
        #         </div>
        #         """, unsafe_allow_html=True)
        #     with col2_stat:
        #         st.markdown("**1000+**")
        #         st.markdown("Medical Resources")
            
        #     # Stat 3
        #     col1_stat, col2_stat = st.columns([0.2, 0.8])
        #     with col1_stat:
        #         st.markdown("""
        #         <div style="
        #             background-color: rgba(155,89,182,0.1);
        #             width: 50px;
        #             height: 50px;
        #             border-radius: 50%;
        #             display: flex;
        #             align-items: center;
        #             justify-content: center;
        #         ">
        #             <span style="font-size: 1.5rem; color: #9b59b6;">⏱️</span>
        #         </div>
        #         """, unsafe_allow_html=True)
        #     with col2_stat:
        #         st.markdown("**24/7**")
        #         st.markdown("Availability")
            
        #     # Testimonial Card using Streamlit
        #     with st.container():
        #         st.markdown("""
        #         <div style="
        #             position: relative;
        #             padding-top: 20px;
        #         ">
        #             <div style="
        #                 position: absolute;
        #                 top: 0;
        #                 left: 20px;
        #                 font-size: 2rem;
        #                 color: #3498db;
        #             ">"</div>
        #             <p style="font-style: italic; margin: 1.5rem 0 1rem 0; padding: 0 1rem;">
        #             This AI system has transformed how we approach complex cases, providing comprehensive insights that complement our clinical expertise.
        #             </p>
        #         </div>
        #         """, unsafe_allow_html=True)
                
        #         # Author info
        #         col1_author, col2_author = st.columns([0.2, 0.8])
        #         with col1_author:
        #             st.markdown("""
        #             <div style="
        #                 background-color: #3498db;
        #                 width: 40px;
        #                 height: 40px;
        #                 border-radius: 50%;
        #                 display: flex;
        #                 align-items: center;
        #                 justify-content: center;
        #                 color: white;
        #                 font-weight: bold;
        #             ">DR</div>
        #             """, unsafe_allow_html=True)
        #         with col2_author:
        #             st.markdown("**Dr. Sarah Johnson**")
        #             st.markdown("*Internal Medicine Specialist*")
        
        # # Close the centered container div
        # st.markdown("</div>", unsafe_allow_html=True)
        
  

elif page == "New Consultation":
    st.markdown("<h1 class='main-header'>New Patient Consultation</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Multi-Agent Medical Intelligence Analysis</p>", unsafe_allow_html=True)
    
    # Patient information form
    with st.container():
        st.subheader("Patient Information")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            patient_name = st.text_input('Patient Name', placeholder='Full Name')
        with col2:
            gender = st.selectbox('Gender', ['Male', 'Female', 'Other'])
        with col3:
            age = st.number_input('Age', min_value=0, max_value=120, value=35)
        with col4:
            weight_kg = st.number_input('Weight (kg)', min_value=0, max_value=500, value=70)
    
    # Tabs for different types of information
    tab1, tab2, tab3, tab4 = st.tabs(["Symptoms & History", "Vital Signs", "Lab Results", "Additional Information"])
    
    with tab1:
        symptoms = st.text_area('Presenting Symptoms', placeholder='e.g., fever, cough, headache, fatigue', height=150)
        symptom_duration = st.text_input('Symptom Duration', placeholder='e.g., 3 days, 2 weeks')
        medical_history = st.text_area('Medical History', placeholder='e.g., diabetes, hypertension, past surgeries', height=150)
        family_history = st.text_input('Family History', placeholder='e.g., heart disease, cancer')
        medications = st.text_area('Current Medications', placeholder='e.g., lisinopril 10mg daily, metformin 500mg twice daily', height=100)
        allergies = st.text_area('Allergies', placeholder='e.g., penicillin, latex, peanuts', height=75)
    
    with tab2:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            temperature = st.number_input('Temperature (°C)', min_value=30.0, max_value=45.0, value=37.0, step=0.1)
        with col2:
            heart_rate = st.number_input('Heart Rate (bpm)', min_value=30, max_value=250, value=75)
        with col3:
            sys_bp = st.number_input('Systolic BP (mmHg)', min_value=50, max_value=250, value=120)
        with col4:
            dia_bp = st.number_input('Diastolic BP (mmHg)', min_value=30, max_value=150, value=80)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            respiratory_rate = st.number_input('Respiratory Rate (breaths/min)', min_value=5, max_value=60, value=16)
        with col2:
            oxygen_saturation = st.number_input('Oxygen Saturation (%)', min_value=50, max_value=100, value=98)
        with col3:
            pain_level = st.slider('Pain Level (0-10)', min_value=0, max_value=10, value=0)
    
    with tab3:
        st.info("Enter any available lab results (optional)")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            hb = st.number_input('Hemoglobin (g/dL)', min_value=0.0, max_value=25.0, value=14.0, step=0.1)
            wbc = st.number_input('WBC (×10^9/L)', min_value=0.0, max_value=50.0, value=7.5, step=0.1)
            platelets = st.number_input('Platelets (×10^9/L)', min_value=0, max_value=1000, value=250)
        with col2:
            glucose = st.number_input('Glucose (mg/dL)', min_value=0, max_value=600, value=100)
            creatinine = st.number_input('Creatinine (mg/dL)', min_value=0.0, max_value=15.0, value=1.0, step=0.1)
            bun = st.number_input('BUN (mg/dL)', min_value=0, max_value=200, value=15)
        with col3:
            sodium = st.number_input('Sodium (mEq/L)', min_value=100, max_value=180, value=140)
            potassium = st.number_input('Potassium (mEq/L)', min_value=2.0, max_value=8.0, value=4.0, step=0.1)
            chloride = st.number_input('Chloride (mEq/L)', min_value=80, max_value=130, value=100)
        
        additional_labs = st.text_area('Additional Lab Results', placeholder='e.g., Liver function tests, Cardiac enzymes', height=100)
    
    with tab4:
        lifestyle = st.text_area('Lifestyle Information', placeholder='e.g., smoker, alcohol consumption, exercise habits', height=100)
        occupation = st.text_input('Occupation', placeholder='e.g., teacher, construction worker')
        recent_travel = st.text_area('Recent Travel', placeholder='e.g., international travel in the last 3 months', height=100)
        exposure_history = st.text_input('Exposure History', placeholder='e.g., sick contacts, environmental exposures')
        additional_notes = st.text_area('Additional Clinical Notes', placeholder='Any other relevant information', height=100)

    # Define Tasks with callbacks to save output
    def save_diagnosis_output(output):
        filename = save_task_output("diagnosis", output)
        st.session_state.task_output_files["diagnosis"] = filename
        return output
    
    diagnose_task = Task(
        description=(
            f"1. Analyze the patient profile: {patient_name}, {gender}, {age} years old, {weight_kg}kg.\n"
            f"2. Review symptoms: {symptoms}\n"
            f"3. Symptom duration: {symptom_duration}\n"
            f"4. Consider medical history: {medical_history}\n"
            f"5. Family history: {family_history}\n"
            f"6. Note current medications: {medications}\n"
            f"7. Be aware of allergies: {allergies}\n"
            f"8. Consider vital signs: Temp {temperature}°C, HR {heart_rate}, BP {sys_bp}/{dia_bp}, RR {respiratory_rate}, O2 Sat {oxygen_saturation}%, Pain level {pain_level}/10\n"
            f"9. Review lab results if available: Hb {hb}, WBC {wbc}, Platelets {platelets}, Glucose {glucose}, Creatinine {creatinine}, BUN {bun}, Sodium {sodium}, Potassium {potassium}, Chloride {chloride}, Additional labs: {additional_labs}\n"
            f"10. Consider lifestyle factors: {lifestyle}\n"
            f"11. Note occupation: {occupation}\n"
            f"12. Recent travel: {recent_travel}\n"
            f"13. Exposure history: {exposure_history}\n"
            f"14. Additional notes: {additional_notes}\n"
            f"15. Based on all the information above, coordinate with the specialist diagnostician as needed to develop a comprehensive differential diagnosis.\n"
            f"16. Prioritize diagnoses based on likelihood and severity.\n"
            f"17. Provide a detailed explanation of your diagnostic reasoning.\n"
            f"18. Format your diagnosis with a clear ## DIAGNOSIS section header."
        ),
        agent=primary_diagnostician,
        expected_output="A comprehensive differential diagnosis with detailed explanation of diagnostic reasoning.",
        callback=lambda output: save_diagnosis_output(output),
        output_file="diagnose_task.md"
    )
    
    def save_specialist_output(output):
        filename = save_task_output("specialist", output)
        st.session_state.task_output_files["specialist"] = filename
        return output
    
    specialist_consult_task = Task(
        description=(
            "1. Review the primary diagnostician's differential diagnosis.\n"
            "2. Provide specialized expertise for any complex conditions identified.\n"
            "3. Evaluate the diagnoses from your specialist perspective.\n"
            "4. Refine or expand the diagnosis based on your specialized knowledge.\n"
            "5. Identify any rare or complex conditions that might have been overlooked.\n"
            "6. Provide probability estimates for each diagnosis in the differential.\n"
            "7. Format your contribution to integrate with the primary diagnosis."
        ),
        agent=specialist_diagnostician,
        expected_output="Specialized diagnostic assessment that refines or validates the primary diagnosis.",
        callback=lambda output: save_specialist_output(output)
    )
    
    def save_treatment_output(output):
        filename = save_task_output("treatment", output)
        st.session_state.task_output_files["treatment"] = filename
        return output
    
    treatment_plan_task = Task(
        description=(
            f"1. Based on the confirmed diagnoses, develop a comprehensive treatment plan for {patient_name}, {gender}, {age} years old, {weight_kg}kg.\n"
            f"2. Consider the patient's current medications: {medications}\n"
            f"3. Consider allergies: {allergies}\n"
            f"4. Consider medical history: {medical_history}\n"
            f"5. Consider vital signs and lab values when determining appropriate treatments.\n"
            f"6. Consult with the pharmacology specialist about medication choices and potential interactions.\n"
            f"7. Include both pharmacological and non-pharmacological interventions.\n"
            f"8. Provide specific medication dosages, frequencies, and durations when applicable.\n"
            f"9. Include follow-up recommendations and monitoring parameters.\n"
            f"10. Format your plan with a clear ## TREATMENT PLAN section header."
        ),
        agent=treatment_advisor,
        expected_output="A comprehensive, individualized treatment plan that addresses all diagnosed conditions.",
        callback=lambda output: save_treatment_output(output)
    )
    
    def save_med_safety_output(output):
        filename = save_task_output("med_safety", output)
        st.session_state.task_output_files["med_safety"] = filename
        return output
    
    medication_safety_task = Task(
        description=(
            f"1. Review the proposed medications in the treatment plan.\n"
            f"2. Check for potential drug interactions with current medications: {medications}\n"
            f"3. Verify appropriate dosages based on patient profile: {age} years, {weight_kg}kg, creatinine {creatinine}.\n"
            f"4. Check for contraindications based on allergies: {allergies}\n"
            f"5. Check for contraindications based on medical history: {medical_history}\n"
            f"6. Recommend dosage adjustments if necessary.\n"
            f"7. Suggest alternative medications if safety issues are identified.\n"
            f"8. Provide key counseling points for each medication.\n"
            f"9. Include your analysis in the safety assessment."
        ),
        agent=pharmacology_specialist,
        expected_output="A medication safety analysis that identifies potential issues and provides recommendations.",
        callback=lambda output: save_med_safety_output(output)
    )
    
    def save_research_output(output):
        filename = save_task_output("research", output)
        st.session_state.task_output_files["research"] = filename
        return output
    
    research_task = Task(
        description=(
            "1. Search for relevant, recent medical literature related to the diagnoses and treatments.\n"
            "2. Identify evidence-based guidelines that support the diagnostic and treatment recommendations.\n"
            "3. Find any recent research that might influence the care plan.\n"
            "4. Evaluate the quality and relevance of the evidence.\n"
            "5. Synthesize the research findings into actionable insights.\n"
            "6. Include 3-5 specific, relevant citations.\n"
            "7. Format your findings with a clear ## MEDICAL RESEARCH section header."
        ),
        agent=medical_researcher,
        expected_output="A synthesis of relevant medical literature that supports the diagnosis and treatment recommendations.",
        callback=lambda output: save_research_output(output)
    )
    
    def save_education_output(output):
        filename = save_task_output("patient_education", output)
        st.session_state.task_output_files["patient_education"] = filename
        return output
    
    patient_education_task = Task(
        description=(
            f"1. Based on the diagnosis and treatment plan, develop educational materials for {patient_name}.\n"
            f"2. Consider the patient's profile: {gender}, {age} years old.\n"
            f"3. Explain the condition(s) in clear, accessible language.\n"
            f"4. Provide information about the prescribed treatments and why they're important.\n"
            f"5. Include lifestyle modifications and self-care strategies.\n"
            f"6. Explain warning signs that should prompt medical attention.\n"
            f"7. Address common questions patients might have about their condition.\n"
            f"8. Create monitoring guidance if applicable.\n"
            f"9. Format your materials with a clear ## PATIENT EDUCATION section header."
        ),
        agent=patient_educator,
        expected_output="Clear, accessible patient education materials tailored to the diagnosis and treatment plan.",
        callback=lambda output: save_education_output(output)
    )
    
    def save_safety_output(output):
        filename = save_task_output("safety", output)
        st.session_state.task_output_files["safety"] = filename
        return output
    
    safety_assessment_task = Task(
        description=(
            "1. Review the entire case, including diagnosis, treatment plan, and medication recommendations.\n"
            "2. Identify any critical safety concerns that require immediate attention.\n"
            "3. Highlight key warning signs that should prompt emergency care.\n"
            "4. Note any diagnosis or treatment risks that clinicians should be aware of.\n"
            "5. Recommend appropriate safety monitoring parameters.\n"
            "6. Suggest precautions to minimize risks.\n"
            "7. Format your assessment with a clear ## ALERTS section header."
        ),
        agent=safety_officer,
        expected_output="A safety assessment that identifies potential risks and provides safety recommendations.",
        callback=lambda output: save_safety_output(output)
    )
    
    # Create Crew with optimized process
    medical_crew = Crew(
        agents=[
            primary_diagnostician,
            specialist_diagnostician,
            treatment_advisor,
            pharmacology_specialist,
            medical_researcher,
            patient_educator,
            safety_officer
        ],
        tasks=[
            diagnose_task,
            specialist_consult_task,
            treatment_plan_task,
            medication_safety_task,
            research_task,
            patient_education_task,
            safety_assessment_task
        ],
        verbose=True,
        process=Process.sequential,  # Use sequential process for medical workflow
        manager_llm=llm
    )
    
    # Run analysis when button is clicked
    if st.button("Run Medical Analysis"):
        if not symptoms:
            st.error("Please enter the patient's symptoms to continue.")
        else:
            # Clear previous task outputs
            st.session_state.task_output_files = {}
            
            # Display progress animation
            progress_animation()
            
            # Run the crew
            with st.spinner("AI Medical Agents are analyzing the case..."):
                result = medical_crew.kickoff()
                
                # Debug: Print task output files
                # st.write("Task output files:", st.session_state.task_output_files)
                
                # Parse the results from saved files
                parsed_results = {}
                for task_name in ["diagnosis", "treatment", "research", "safety", "patient_education"]:
                    if task_name in st.session_state.task_output_files:
                        try:
                            with open(st.session_state.task_output_files[task_name], 'r', encoding='utf-8') as f:
                                content = f.read()
                                parsed_results[task_name] = content
                        except Exception as e:
                            st.error(f"Error reading {task_name} file: {str(e)}")
                            parsed_results[task_name] = f"Error loading {task_name} content"
                
                # Store consultation in session state
                new_consultation = {
                    "id": len(st.session_state.past_consultations) + 1,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "patient_name": patient_name if patient_name else "Anonymous Patient",
                    "age": age,
                    "gender": gender,
                    "main_symptoms": symptoms,
                    "results": parsed_results,
                    "task_files": st.session_state.task_output_files
                }
                st.session_state.past_consultations.append(new_consultation)
            
            # Display results
            st.success("Medical analysis complete!")
            
            # Debug: Print parsed results

            
            # Display all sections that have content
            sections = [
                ("diagnosis", "🔍 Diagnosis", "#f8f9fa", "#3498db"),
                ("treatment", "💊 Treatment Plan", "#e8f4f8", "#2ecc71"),
                ("research", "📚 Supporting Medical Research", "#f0f7ee", "#27ae60"),
                ("safety", "⚠️ Important Alerts", "#fde9e8", "#e74c3c"),
                ("patient_education", "📋 Patient Education", "#e8f4fd", "#3498db")
            ]
            
            for section in sections:
                key, title, bg_color, border_color = section
                if key in parsed_results and parsed_results[key]:
                    with st.container():
                        st.markdown(f"""
                        <div class="card" style="background-color: {bg_color}; border-left: 5px solid {border_color};">
                            <h3 style="color: #2c3e50;">{title}</h3>
                            <div style="padding: 10px;">
                        """, unsafe_allow_html=True)
                        st.markdown(parsed_results[key])
                        st.markdown("</div></div>", unsafe_allow_html=True)
            
            # Generate prescription if treatment plan exists
            if "treatment" in parsed_results and parsed_results["treatment"]:
                extracted_medications = extract_medications(parsed_results["treatment"])
                main_diagnosis = ""
                if "diagnosis" in parsed_results and parsed_results["diagnosis"]:
                    main_diagnosis = parsed_results["diagnosis"].split("\n")[0] if "\n" in parsed_results["diagnosis"] else parsed_results["diagnosis"]
                
                prescription_html = generate_prescription(
                    patient_name if patient_name else "Anonymous Patient",
                    gender,
                    age,
                    main_diagnosis[:100],
                    extracted_medications
                )
                
                # Display buttons for additional actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("View Prescription"):
                        st.markdown(prescription_html, unsafe_allow_html=True)
                with col2:
                    if st.button("Download Full Report"):
                        report = "# Medical Consultation Report\n\n"
                        report += f"## Patient Information\n- Name: {patient_name if patient_name else 'Anonymous Patient'}\n"
                        report += f"- Age: {age} years\n- Gender: {gender}\n"
                        report += f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                        
                        for section in sections:
                            key, title, _, _ = section
                            if key in parsed_results and parsed_results[key]:
                                report += f"## {title}\n{parsed_results[key]}\n\n"
                        
                        b64 = base64.b64encode(report.encode()).decode()
                        href = f'<a href="data:file/txt;base64,{b64}" download="medical_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt">Click here to download</a>'
                        st.markdown(href, unsafe_allow_html=True)
                with col3:
                    if st.button("Start New Consultation"):
                        st.session_state.current_page = "New Consultation"
                        st.rerun()


elif page == "Past Consultations":
    st.markdown("<h1 class='main-header'>Past Consultations</h1>", unsafe_allow_html=True)
    
    if not st.session_state.past_consultations:
        st.info("No past consultations found. Start a new consultation to see results here.")
    else:
        # Create a dataframe for easier display
        consultations_df = pd.DataFrame([
            {
                "ID": c["id"],
                "Date": c["timestamp"],
                "Patient": c["patient_name"],
                "Age": c["age"],
                "Gender": c["gender"],
                "Symptoms": c["main_symptoms"][:50] + "..." if len(c["main_symptoms"]) > 50 else c["main_symptoms"]
            } for c in st.session_state.past_consultations
        ])
        
        # Display as a table
        st.dataframe(consultations_df, use_container_width=True)
        
        # Allow selection of consultation to view details
        selected_id = st.selectbox("Select consultation to view details", 
                                 options=consultations_df["ID"].tolist(),
                                 format_func=lambda x: f"Consultation #{x} - {consultations_df[consultations_df['ID']==x]['Patient'].values[0]} ({consultations_df[consultations_df['ID']==x]['Date'].values[0]})")
        
        if selected_id:
            # Find the selected consultation
            selected_consultation = next((c for c in st.session_state.past_consultations if c["id"] == selected_id), None)
            
            if selected_consultation:
                # Display consultation details
                st.subheader(f"Consultation for {selected_consultation['patient_name']}")
                st.markdown(f"**Date:** {selected_consultation['timestamp']}")
                st.markdown(f"**Patient:** {selected_consultation['patient_name']}, {selected_consultation['age']} years, {selected_consultation['gender']}")
                
                # Display the content in tabs with proper card styling
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Diagnosis", "Treatment", "Research", "Alerts", "Patient Education"])
                
                with tab1:
                    st.markdown("""
                    <div class="card">
                        <h3>🔍 Differential Diagnosis</h3>
                        <div class="markdown-text-container">
                    """, unsafe_allow_html=True)
                    st.markdown(selected_consultation["results"].get("diagnosis", "No diagnosis available"))
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with tab2:
                    st.markdown("""
                    <div class="card">
                        <h3>💊 Treatment Plan</h3>
                        <div class="markdown-text-container">
                    """, unsafe_allow_html=True)
                    st.markdown(selected_consultation["results"].get("treatment", "No treatment plan available"))
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with tab3:
                    st.markdown("""
                    <div class="card">
                        <h3>📚 Medical Research</h3>
                        <div class="markdown-text-container">
                    """, unsafe_allow_html=True)
                    st.markdown(selected_consultation["results"].get("research", "No research available"))
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with tab4:
                    st.markdown("""
                    <div class="card">
                        <h3>⚠️ Important Alerts</h3>
                        <div class="markdown-text-container">
                    """, unsafe_allow_html=True)
                    # Check both possible locations for alerts
                    alerts_content = selected_consultation["results"].get("alerts", 
                                      selected_consultation["results"].get("safety", "No critical alerts for this consultation."))
                    st.markdown(alerts_content)
                    st.markdown("</div></div>", unsafe_allow_html=True)
                
                with tab5:
                    st.markdown("""
                    <div class="card">
                        <h3>📋 Patient Education</h3>
                        <div class="markdown-text-container">
                    """, unsafe_allow_html=True)
                    st.markdown(selected_consultation["results"].get("patient_education", "No patient education materials available"))
                    st.markdown("</div></div>", unsafe_allow_html=True)

elif page == "Medical Knowledge":
    st.markdown("<h1 class='main-header'>Medical Knowledge Database</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        This knowledge base is powered by specialized AI agents that can search, analyze, and summarize medical information from trusted sources.
    </div>
    """, unsafe_allow_html=True)

    # Define specialized agents for medical knowledge
    medical_research_agent = Agent(
        role="Medical Research Specialist",
        goal="Find and summarize the most relevant medical information from trusted sources",
        backstory="""You are an expert medical researcher with access to the latest clinical guidelines, 
        research papers, and medical databases. You can quickly find and synthesize accurate medical 
        information on any topic.""",
        llm=llm,
        verbose=True,
        allow_delegation=False,
        max_iter=2,
        tools=[search_tool, scrape_tool]
    )

    evidence_evaluator = Agent(
        role="Medical Evidence Evaluator",
        goal="Evaluate the quality and relevance of medical evidence",
        backstory="""You are a critical appraiser of medical evidence with expertise in evidence-based 
        medicine. You assess the reliability, validity, and clinical relevance of medical information.""",
        llm=llm,
        verbose=True,
        max_iter=2,
        allow_delegation=False
    )

    knowledge_synthesizer = Agent(
        role="Medical Knowledge Synthesizer",
        goal="Create clear, organized summaries of medical information",
        backstory="""You specialize in transforming complex medical information into clear, structured 
        knowledge that's easy to understand. You organize information logically and highlight key points.""",
        llm=llm,
        verbose=True,
        max_iter=2,
        allow_delegation=False
    )

    # Search interface
    st.subheader("Search Medical Knowledge")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        search_term = st.text_input("Enter medical term or condition", placeholder="e.g., hypertension, diabetes, asthma")
    with col2:
        st.write("")
        st.write("")
        search_button = st.button("Search")
    
    if search_button and search_term:
        with st.spinner(f"Searching for information about '{search_term}'..."):
            # Define tasks for the knowledge search workflow
            research_task = Task(
                description=f"""
                1. Search for comprehensive information about: {search_term}
                2. Include information from authoritative medical sources
                3. Find details about:
                   - Definition and pathophysiology
                   - Clinical presentation
                   - Diagnostic criteria
                   - Treatment guidelines
                   - Recent research findings
                4. Return raw search results with sources
                """,
                agent=medical_research_agent,
                expected_output=f"A comprehensive collection of information about {search_term} from medical sources"
            )

            evaluation_task = Task(
                description=f"""
                1. Review the research findings about {search_term}
                2. Evaluate the quality of each source
                3. Identify the most reliable and relevant information
                4. Filter out outdated or low-quality sources
                5. Highlight any controversies or conflicting evidence
                """,
                agent=evidence_evaluator,
                expected_output=f"An evaluation of the quality and reliability of information about {search_term}",
                context=[research_task]
            )

            synthesis_task = Task(
                description=f"""
                1. Organize the validated information about {search_term} into clear sections with these exact headers:
                   - ## DEFINITION: Definition and Overview
                   - ## CLINICAL PRESENTATION: Clinical Presentation
                   - ## DIAGNOSTIC APPROACH: Diagnostic Approach
                   - ## TREATMENT OPTIONS: Treatment Options
                   - ## RECENT ADVANCES: Recent Advances
                   - ## REFERENCES: Key References
                
                2. For each section:
                   - Start with a concise 2-3 sentence summary
                   - Use bullet points for key features (- feature)
                   - Bold important terms (**term**)
                   - Separate concepts with blank lines
                   - Keep paragraphs short (max 3 sentences)
                
                3. Example format:
                   ## DEFINITION: 
                   [Brief 1-2 sentence definition]
                   
                   **Key Characteristics:**
                   - Characteristic 1
                   - Characteristic 2
                   - Characteristic 3
                   
                   [Additional details in short paragraphs]
                
                4. Use professional but accessible language
                """,
                agent=knowledge_synthesizer,
                expected_output=f"A well-structured medical knowledge summary with clear sectioning and formatting",
                context=[evaluation_task]
            )

            # Create and run the knowledge crew
            knowledge_crew = Crew(
                agents=[medical_research_agent, evidence_evaluator, knowledge_synthesizer],
                tasks=[research_task, evaluation_task, synthesis_task],
                verbose=True,
                process=Process.sequential
            )

            result = knowledge_crew.kickoff()

            # Function to parse sections from the result
            def parse_knowledge_sections(content):
                sections = {
                    "definition": "",
                    "clinical_presentation": "",
                    "diagnostic_approach": "",
                    "treatment_options": "",
                    "recent_advances": "",
                    "references": ""
                }
                
                if not isinstance(content, str):
                    content = str(content)
                
                # Parse definition
                if "## DEFINITION:" in content:
                    sections["definition"] = content.split("## DEFINITION:")[1].split("## CLINICAL PRESENTATION:")[0].strip()
                
                # Parse clinical presentation
                if "## CLINICAL PRESENTATION:" in content:
                    sections["clinical_presentation"] = content.split("## CLINICAL PRESENTATION:")[1].split("## DIAGNOSTIC APPROACH:")[0].strip()
                
                # Parse diagnostic approach
                if "## DIAGNOSTIC APPROACH:" in content:
                    sections["diagnostic_approach"] = content.split("## DIAGNOSTIC APPROACH:")[1].split("## TREATMENT OPTIONS:")[0].strip()
                
                # Parse treatment options
                if "## TREATMENT OPTIONS:" in content:
                    sections["treatment_options"] = content.split("## TREATMENT OPTIONS:")[1].split("## RECENT ADVANCES:")[0].strip()
                
                # Parse recent advances
                if "## RECENT ADVANCES:" in content:
                    sections["recent_advances"] = content.split("## RECENT ADVANCES:")[1].split("## REFERENCES:")[0].strip()
                
                # Parse references
                if "## REFERENCES:" in content:
                    sections["references"] = content.split("## REFERENCES:")[1].strip()
                
                return sections

            # Parse the results into sections
            knowledge_sections = parse_knowledge_sections(result)

            # Custom CSS for knowledge cards
            st.markdown("""
            <style>
            .knowledge-card {
                background-color: white;
                border-radius: 10px;
                padding: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                margin-bottom: 20px;
                border-left: 5px solid #3498db;
                transition: transform 0.3s ease;
            }
            .knowledge-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
            }
            .knowledge-card h3 {
                color: #2c3e50;
                margin-top: 0;
                padding-bottom: 10px;
                border-bottom: 1px solid #eee;
            }
            .knowledge-content {
                line-height: 1.6;
                font-size: 15px;
            }
            .knowledge-content p {
                margin-bottom: 12px;
            }
            .knowledge-content ul {
                padding-left: 25px;
                margin-bottom: 15px;
            }
            .knowledge-content li {
                margin-bottom: 8px;
                list-style-type: disc;
            }
            .knowledge-content strong {
                color: #2c3e50;
                font-weight: 600;
            }
            .definition-card {
                border-left-color: #3498db;
            }
            .clinical-card {
                border-left-color: #2ecc71;
            }
            .diagnostic-card {
                border-left-color: #f39c12;
            }
            .treatment-card {
                border-left-color: #9b59b6;
            }
            .advances-card {
                border-left-color: #e74c3c;
            }
            .references-card {
                border-left-color: #1abc9c;
            }
            .knowledge-header {
                display: flex;
                align-items: center;
                margin-bottom: 15px;
            }
            .knowledge-icon {
                font-size: 24px;
                margin-right: 10px;
            }
            </style>
            """, unsafe_allow_html=True)

            # Display results
            st.success("Medical knowledge search complete!")

            # Display in expandable sections with enhanced styling
            with st.expander("📖 Definition and Overview", expanded=True):
                if knowledge_sections["definition"]:
                    st.markdown(f"""
                    <div class="knowledge-card definition-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">📖</span>
                            <h3>Definition and Overview</h3>
                        </div>
                        <div class="knowledge-content">
                            {knowledge_sections["definition"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="knowledge-card definition-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">📖</span>
                            <h3>Definition and Overview</h3>
                        </div>
                        <p>No definition information available for this term.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("🩺 Clinical Presentation"):
                if knowledge_sections["clinical_presentation"]:
                    st.markdown(f"""
                    <div class="knowledge-card clinical-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">🩺</span>
                            <h3>Clinical Presentation</h3>
                        </div>
                        <div class="knowledge-content">
                            {knowledge_sections["clinical_presentation"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="knowledge-card clinical-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">🩺</span>
                            <h3>Clinical Presentation</h3>
                        </div>
                        <p>No clinical presentation information available.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("🔍 Diagnostic Approach"):
                if knowledge_sections["diagnostic_approach"]:
                    st.markdown(f"""
                    <div class="knowledge-card diagnostic-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">🔍</span>
                            <h3>Diagnostic Approach</h3>
                        </div>
                        <div class="knowledge-content">
                            {knowledge_sections["diagnostic_approach"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="knowledge-card diagnostic-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">🔍</span>
                            <h3>Diagnostic Approach</h3>
                        </div>
                        <p>No diagnostic approach information available.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("💊 Treatment Options"):
                if knowledge_sections["treatment_options"]:
                    st.markdown(f"""
                    <div class="knowledge-card treatment-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">💊</span>
                            <h3>Treatment Options</h3>
                        </div>
                        <div class="knowledge-content">
                            {knowledge_sections["treatment_options"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="knowledge-card treatment-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">💊</span>
                            <h3>Treatment Options</h3>
                        </div>
                        <p>No treatment options information available.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("🚀 Recent Advances"):
                if knowledge_sections["recent_advances"]:
                    st.markdown(f"""
                    <div class="knowledge-card advances-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">🚀</span>
                            <h3>Recent Advances</h3>
                        </div>
                        <div class="knowledge-content">
                            {knowledge_sections["recent_advances"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="knowledge-card advances-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">🚀</span>
                            <h3>Recent Advances</h3>
                        </div>
                        <p>No recent advances information available.</p>
                    </div>
                    """, unsafe_allow_html=True)

            with st.expander("📚 References"):
                if knowledge_sections["references"]:
                    st.markdown(f"""
                    <div class="knowledge-card references-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">📚</span>
                            <h3>References</h3>
                        </div>
                        <div class="knowledge-content">
                            {knowledge_sections["references"]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="knowledge-card references-card">
                        <div class="knowledge-header">
                            <span class="knowledge-icon">📚</span>
                            <h3>References</h3>
                        </div>
                        <p>No references available.</p>
                    </div>
                    """, unsafe_allow_html=True)

            # Visualization of search results (mock data)
            st.subheader("Knowledge Graph")
            categories = ["Definition", "Diagnosis", "Treatment", "Research"]
            confidence = [90, 85, 80, 75]  # Mock confidence scores
            
            fig = go.Figure(go.Bar(
                x=categories,
                y=confidence,
                marker_color=['#3498db', '#2ecc71', '#f39c12', '#9b59b6']
            ))
            fig.update_layout(
                title="Information Confidence Levels",
                yaxis_title="Confidence Score (%)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    else:
        # Show knowledge base statistics when no search is active
        pass

# elif page == "Settings":
#     st.markdown("<h1 class='main-header'>MediAssist AI Settings</h1>", unsafe_allow_html=True)
    
#     st.subheader("API Configuration")
    
#     serper_key = st.text_input("Serper API Key", value=os.getenv("SERPER_API_KEY", ""), type="password")
#     google_key = st.text_input("Google API Key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")
    
#     if st.button("Save API Settings"):
#         # In a real app, these would be securely stored
#         os.environ["SERPER_API_KEY"] = serper_key
#         os.environ["GOOGLE_API_KEY"] = google_key
#         st.success("API settings saved!")
    
#     st.divider()
    
#     st.subheader("Agent Settings")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.markdown("#### Model Selection")
#         model_option = st.selectbox(
#             "Select AI Model for Agents",
#             ["gemini/gemini-1.5-flash", "gemini/gemini-1.5-pro", "gpt-4", "claude-3-opus"]
#         )
        
#         st.markdown("#### Process Type")
#         process_type = st.radio(
#             "Agent Interaction Process",
#             ["Sequential", "Hierarchical"],
#             index=0
#         )
    
#     with col2:
#         st.markdown("#### Agent Verbosity")
#         verbose_setting = st.checkbox("Enable Verbose Mode", value=True)
        
#         st.markdown("#### Response Format")
#         format_preference = st.radio(
#             "Preferred Response Format",
#             ["Comprehensive", "Concise", "Technical", "Patient-Friendly"],
#             index=0
#         )
    
#     st.divider()
    
#     st.subheader("Application Settings")
    
#     col1, col2 = st.columns(2)
    
#     with col1:
#         theme = st.selectbox(
#             "Application Theme",
#             ["Light", "Dark", "System Default"]
#         )
        
#         language = st.selectbox(
#             "Language",
#             ["English", "Spanish", "French", "German", "Chinese"]
#         )
    
#     with col2:
#         save_history = st.checkbox("Save Consultation History", value=True)
#         auto_refresh = st.checkbox("Auto-refresh Knowledge Base", value=True)
    
#     if st.button("Apply Settings"):
#         st.success("Settings applied successfully!")

elif page == "About":
    st.markdown("<h1 class='main-header'>About MediAssist AI</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Multi-Agent AI Medical Assistant Platform
        
        MediAssist AI is an advanced platform that leverages multiple specialized AI agents to provide comprehensive medical decision support. By combining the expertise of various specialized agents, the system delivers more thorough and accurate medical assessments than a single AI model could provide.
        
        #### Core Technology
        The platform is built on a collaborative multi-agent framework, where each agent has specialized medical knowledge and responsibilities:
        
        - **Primary Diagnostician**: Coordinates the initial diagnostic process
        - **Specialist Diagnostician**: Provides specialized medical expertise
        - **Treatment Advisor**: Develops personalized treatment strategies
        - **Pharmacology Specialist**: Ensures medication safety and efficacy
        - **Medical Researcher**: Integrates current medical literature
        - **Patient Educator**: Creates accessible educational materials
        - **Safety Officer**: Identifies potential risks and precautions
        - **medical_research_agent**: summarize the most relevant medical information from trusted sources
        - **evidence_evaluator**: Evaluate the quality and relevance of medical evidence
        - **knowledge_synthesizer**: Create clear, organized summaries of medical information
        
        #### How It Works
        The agents collaborate through a structured workflow, sharing information and building upon each other's expertise to develop a comprehensive medical assessment and plan.
        
        
        """)
        

        st.markdown("""
        <div class="card" style="border-left: 5px solid #e74c3c; background-color: #fde9e8;">
            <h3 style="color: #e74c3c;">⚠️ Important Disclaimer</h3>
            <div class="markdown-text-container">
                <strong>This application is for educational and demonstration purposes only.</strong> It is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified healthcare providers with any questions you may have regarding medical conditions.
                <br><br>
                <strong>Important Note:</strong> This is a demonstration of AI capabilities in healthcare. The system is not FDA-approved for clinical use and should not be used for actual medical decision-making.
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        # Display a mock architecture diagram
        
        
        st.markdown("#### Version Information")
        st.markdown("- **AI Framework:** CrewAI")
        st.markdown("- **Last Updated:** March 2025")
        
        st.markdown("#### Development")
        st.markdown("Developed as a demonstration of multi-agent AI systems in healthcare applications.")
    
    # Contact form
    st.subheader("Contact")
    
    with st.form("contact_form"):
        st.markdown("Have questions or feedback about MediAssist AI?")
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Name")
        with col2:
            email = st.text_input("Email")
        
        message = st.text_area("Message")
        
        submitted = st.form_submit_button("Send Message")
        if submitted:
            st.success("Thank you for your message! This is a demo form and does not actually send messages.")

# Run the application
if __name__ == "__main__":
    pass



