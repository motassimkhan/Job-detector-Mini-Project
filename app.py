import streamlit as st
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd
from io import StringIO
import plotly.express as px
import plotly.graph_objects as go

# Import core job verification system
try:
    from functions import (
        UnifiedVerificationService, 
        JobPostResult, 
        CompanyInfo,
        setup_environment
    )
except ImportError:
    st.error("⚠️ Cannot import job_core.py. Make sure the file exists in the same directory.")
    st.stop()

# page configuration

st.set_page_config(
    page_title="Job Verification Service",
    page_icon="🔍",
    layout="wide"
)

# custom css styling

st.markdown("""
<style>
 body {
        background: linear-gradient(135deg, #7b2ff7 0%, #f107a3 100%);
        color: #f0f0f0;
    }
    .stApp {
        background: linear-gradient(135deg, #7b2ff7 0%, #f107a3 100%);
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #2c3e50; /* dark blue-gray */
        margin-bottom: 2rem;
    }

    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2d3436; /* dark grey */
        margin: 1rem 0;
    }

    .status-box {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid;
    }

    .status-real {
        background-color: #e6f4ea; /* soft green */
        border-color: #2e7d32;     /* dark green */
        color: #1b5e20;
    }

    .status-fake {
        background-color: #fdecea; /* soft red */
        border-color: #c62828;     /* dark red */
        color: #b71c1c;
    }

    .status-unknown {
        background-color: #fff9e6; /* light yellow */
        border-color: #f9a825;     /* dark yellow */
        color: #795548;
    }

    .metric-container {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: #212121;
    }

    .company-info {
        background-color: linear-gradient(135deg, #7b2ff7 0%, #f107a3 100%); /* soft blue */
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #aed6f1;
        color: #1b2631;
    }

    .warning-box {
        background-color: #fff9e6;
        border: 1px solid #f7dc6f;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #6d4c41;
    }

    .success-box {
        background-color: #e6f4ea;
        border: 1px solid #a5d6a7;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #000000;
    }
</style>
""", unsafe_allow_html=True)


# utility functions

@st.cache_resource
def initialize_service():
    """Initialize the verification service with caching"""
    try:
        if not setup_environment():
            return None, "Environment setup failed. Please check your API keys."
        
        service = UnifiedVerificationService()
        return service, None
    except Exception as e:
        return None, f"Failed to initialize service: {str(e)}"

def display_verification_result(result: JobPostResult):
    """Display job verification result with styling"""
    if result.label.lower() == 'real':
        status_class = "status-real"
        icon = "✅"
    elif result.label.lower() == 'fake':
        status_class = "status-fake"
        icon = "❌"
    else:
        status_class = "status-unknown"
        icon = "⚠️"
    
    st.markdown(f"""
    <div class="status-box {status_class}">
        <h3>{icon} Classification: {result.label}</h3>
        <p><strong>Analysis:</strong> {result.reasoning}</p>
    </div>
    """, unsafe_allow_html=True)

def display_company_info(company: CompanyInfo):
    """Display company information in a formatted way"""
    st.markdown("""
    <div class="company-info">
        <h3>🏢 Company Information</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Company Name:** {company.name}")
        if company.founding_date:
            st.write(f"**Founded:** {company.founding_date}")
        if company.headquarters:
            st.write(f"**Headquarters:** {company.headquarters}")
    
    with col2:
        if company.industry:
            st.write(f"**Industry:** {company.industry}")
        if company.size:
            st.write(f"**Company Size:** {company.size}")
        if company.website_link:
            st.write(f"**Website:** [Visit Website]({company.website_link})")
    
    if company.summary:
        st.write("**Company Summary:**")
        st.write(company.summary)

def save_results_to_session(result_type: str, data: Dict[str, Any]):
    """Save results to session state for history tracking"""
    if 'verification_history' not in st.session_state:
        st.session_state.verification_history = []
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    history_entry = {
        'timestamp': timestamp,
        'type': result_type,
        'data': data
    }
    
    st.session_state.verification_history.append(history_entry)
    
    # Keep only last 50 entries
    if len(st.session_state.verification_history) > 50:
        st.session_state.verification_history = st.session_state.verification_history[-50:]

def create_verification_chart():
    """Create a chart showing verification history"""
    if 'verification_history' not in st.session_state or not st.session_state.verification_history:
        return None
    
    # Count labels
    labels = []
    for entry in st.session_state.verification_history:
        if entry['type'] == 'job_verification':
            labels.append(entry['data']['job_verification']['label'])
    
    if not labels:
        return None
    
    # Create pie chart
    label_counts = pd.Series(labels).value_counts()
    
    fig = px.pie(
        values=label_counts.values,
        names=label_counts.index,
        title="Job Verification Results Distribution",
        color_discrete_map={
            'Real': '#28a745',
            'Fake': '#dc3545',
            'Unknown': '#ffc107'
        }
    )
    
    return fig

# sidebar 

def setup_sidebar():
    """Setup sidebar with API configuration and settings"""
    st.sidebar.header("He There 👋 ")
    
    # API Keys Section
    with st.sidebar.expander("🔑 API Keys", expanded=False):
        st.markdown("**Required:**")
        groq_key = st.text_input(
            "Groq API Key", 
            type="password", 
            help="Required for LLM operations",
            value=st.secrets['GROQ_API_KEY']
        )
        
        st.markdown("**Optional (for search):**")
        search_api_key = st.text_input(
            "SearchAPI Key", 
            type="password",
            value=st.secrets['SEARCHAPI_KEY']
        )
        
        google_api_key = st.text_input(
            "Google API Key", 
            type="password",
            value=st.secrets['GOOGLE_API_KEY']
        )
        
        google_cse_id = st.text_input(
            "Google CSE ID",
            value=st.secrets['GOOGLE_CSE_ID']
        )
        
        # Set environment variables if provided
        if groq_key:
            st.secrets['GROQ_API_KEY'] = groq_key
        if search_api_key:
            st.secrets['SEARCHAPI_KEY'] = search_api_key
        if google_api_key:
            st.secrets['GOOGLE_API_KEY'] = google_api_key
        if google_cse_id:
            st.secrets['GOOGLE_CSE_ID'] = google_cse_id
    
    # Settings Section
    with st.sidebar.expander("🎛️ Settings", expanded=True):
        st.session_state.auto_research_company = st.checkbox(
            "Auto-research company", 
            value=True,
            help="Automatically research company when verifying job posts"
        )
        
        st.session_state.show_raw_response = st.checkbox(
            "Show raw LLM response", 
            value=False,
            help="Display the raw response from the LLM"
        )
        
        st.session_state.detailed_analysis = st.checkbox(
            "Detailed analysis mode", 
            value=True,
            help="Provide more detailed reasoning in verification"
        )
    
    # Quick Stats
    if 'verification_history' in st.session_state and st.session_state.verification_history:
        st.sidebar.header("📊 Quick Stats")
        total_verifications = len([e for e in st.session_state.verification_history if e['type'] == 'job_verification'])
        st.sidebar.metric("Total Verifications", total_verifications)
        
        if total_verifications > 0:
            fake_count = len([e for e in st.session_state.verification_history 
                            if e['type'] == 'job_verification' and 
                            e['data']['job_verification']['label'] == 'Fake'])
            fake_percentage = (fake_count / total_verifications) * 100
            st.sidebar.metric("Fake Job Posts", f"{fake_percentage:.1f}%")

# MAIN APPLICATION

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Job Post Verification Center</h1>', unsafe_allow_html=True)
    st.markdown("Detect fake job postings and research companies using AI-powered analysis")
    
    # Setup sidebar
    # setup_sidebar()
    
    # Initialize service
    service, error = initialize_service()
    
    if error:
        st.error(f"❌ {error}")
        st.markdown("""
        <div class="warning-box">
            <h4>Setup Required:</h4>
            <p>Please configure your API keys in the sidebar to get started:</p>
            <ul>
                <li><strong>Groq API Key:</strong> Required for AI analysis</li>
                <li><strong>SearchAPI Key or Google API Keys:</strong> Optional, for enhanced research</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Job Verification", 
        "🏢 Company Research", 
        "📊 Comprehensive Analysis",
        "📈 History & Analytics"
    ])
    
    # Tab 1: Job verification
    with tab1:
        st.markdown('<h2 class="sub-header">Job Post Verification</h2>', unsafe_allow_html=True)
        
        # Input methods
        input_method = st.radio(
            "Choose input method:",
            ["📝 Text Input", "📄 File Upload"],
            horizontal=True
        )
        
        job_text = ""
        
        if input_method == "📝 Text Input":
            job_text = st.text_area(
                "Paste the job posting here:",
                height=300,
                placeholder="Paste the complete job posting text here..."
            )
        
        else:  # File Upload
            uploaded_file = st.file_uploader(
                "Upload job posting file",
                type=['txt', 'doc', 'docx', 'pdf'],
                help="Upload a text file containing the job posting"
            )
            
            if uploaded_file is not None:
                try:
                    # Handle different file types
                    if uploaded_file.type == "text/plain":
                        job_text = str(uploaded_file.read(), "utf-8")
                    else:
                        st.warning("Currently only .txt files are supported. Please copy and paste the text instead.")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Verification button and processing
        if st.button("🔍 Verify Job Post", type="primary", use_container_width=True):
            if not job_text.strip():
                st.warning("⚠️ Please enter a job posting to verify.")
            else:
                with st.spinner("🤖 Analyzing job posting..."):
                    try:
                        result = service.verify_job_post(job_text)
                        
                        # Display results
                        display_verification_result(result)
                        
                        # Show raw response if enabled
                        if st.session_state.get('show_raw_response', False):
                            with st.expander("🔧 Raw LLM Response"):
                                st.text(result.raw_response)
                        
                        # Save to history
                        save_results_to_session('job_verification', {
                            'job_verification': {
                                'label': result.label,
                                'reasoning': result.reasoning,
                                'confidence': result.confidence
                            },
                            'job_text_preview': job_text[:200] + "..." if len(job_text) > 200 else job_text
                        })
                        
                        # Success message
                        st.markdown("""
                        <div class="success-box">
                            ✅ Analysis complete! Check the results above.
                        </div>
                        """, unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error during verification: {str(e)}")
        
        # Example job posts
        with st.expander("📋 Example Job Posts"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🟢 Example of a Real Job Post:**")
                real_example = """Software Engineer - TechCorp Inc.

Location: San Francisco, CA
Salary: $90,000 - $120,000

We are seeking a skilled Software Engineer to join our development team.

Responsibilities:
- Develop and maintain web applications using Python/Django
- Collaborate with product managers and designers
- Write clean, testable code
- Participate in code reviews

Requirements:
- Bachelor's degree in Computer Science or related field
- 3+ years of experience in software development
- Proficiency in Python, JavaScript, SQL
- Experience with version control (Git)

Benefits:
- Health, dental, and vision insurance
- 401(k) with company match
- Flexible PTO
- Professional development budget

Contact: careers@techcorp.com
Apply at: https://techcorp.com/careers"""
                
                if st.button("Use Real Example", key="real_ex"):
                    st.session_state.example_job = real_example
            
            with col2:
                st.markdown("**🔴 Example of a Suspicious Job Post:**")
                fake_example = """URGENT!!! Work From Home - Make $5000/week!!!

NO EXPERIENCE NEEDED! IMMEDIATE START!

We need people to work from home doing simple data entry. Earn $5000+ per week guaranteed!

Requirements:
- Must be over 18
- Have computer and internet
- Need $200 processing fee upfront

This is a LIMITED TIME OFFER! Only 5 spots left!

Contact immediately: fastmoney@gmail.com
Send $200 via PayPal to secure your position!

WARNING: This offer expires in 24 hours!!!"""
                
                if st.button("Use Suspicious Example", key="fake_ex"):
                    st.session_state.example_job = fake_example
        
        # Use example if selected
        if 'example_job' in st.session_state:
            st.text_area("Example loaded:", value=st.session_state.example_job, key="example_display")
            if st.button("Clear Example"):
                del st.session_state.example_job
    
    # Tab 2: company research    
    with tab2:
        st.markdown('<h2 class="sub-header">Company Research</h2>', unsafe_allow_html=True)
        
        company_name = st.text_input(
            "Enter company name to research:",
            placeholder="e.g., Google, Microsoft, Apple"
        )
        
        if st.button("🔎 Research Company", type="primary", use_container_width=True):
            if not company_name.strip():
                st.warning("⚠️ Please enter a company name to research.")
            else:
                with st.spinner(f"🔍 Researching {company_name}..."):
                    try:
                        company_info = service.research_company(company_name)
                        
                        # Display company information
                        display_company_info(company_info)
                        
                        # Save to history
                        save_results_to_session('company_research', {
                            'company_info': {
                                'name': company_info.name,
                                'summary': company_info.summary,
                                'founding_date': company_info.founding_date,
                                'headquarters': company_info.headquarters,
                                'industry': company_info.industry,
                                'size': company_info.size,
                                'website_link': company_info.website_link
                            }
                        })
                        
                    except Exception as e:
                        st.error(f"❌ Error during company research: {str(e)}")
    
    # Tab 3: comprehensive analysis    
    with tab3:
        st.markdown('<h2 class="sub-header">Comprehensive Job Analysis</h2>', unsafe_allow_html=True)
        st.markdown("Get both job verification and company research in one analysis.")
        
        comprehensive_job_text = st.text_area(
            "Enter job posting for comprehensive analysis:",
            height=250,
            placeholder="Paste the complete job posting here for full analysis..."
        )
        
        if st.button("🚀 Run Comprehensive Analysis", type="primary", use_container_width=True):
            if not comprehensive_job_text.strip():
                st.warning("⚠️ Please enter a job posting for analysis.")
            else:
                with st.spinner("🤖 Running comprehensive analysis..."):
                    try:
                        results = service.comprehensive_verification(comprehensive_job_text)
                        
                        # Display job verification results
                        st.markdown("### 🔍 Job Verification Results")
                        job_result = JobPostResult(
                            reasoning=results['job_verification']['reasoning'],
                            label=results['job_verification']['label'],
                            confidence=results['job_verification']['confidence']
                        )
                        display_verification_result(job_result)
                        
                        # Display company research if available
                        if results.get('company_research'):
                            st.markdown("### 🏢 Company Research Results")
                            company_data = results['company_research']
                            company_info = CompanyInfo(
                                name=company_data['name'],
                                summary=company_data['summary'],
                                founding_date=company_data['details']['founding_date'],
                                headquarters=company_data['details']['headquarters'],
                                industry=company_data['details']['industry'],
                                size=company_data['details']['size'],
                                website_link=company_data['details']['website']
                            )
                            display_company_info(company_info)
                        else:
                            st.info("ℹ️ No company information could be extracted from the job posting.")
                        
                        # Export results option
                        st.markdown("### 📥 Export Results")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            json_results = json.dumps(results, indent=2)
                            st.download_button(
                                "📄 Download as JSON",
                                data=json_results,
                                file_name=f"job_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
                        
                        with col2:
                            # Create summary report
                            summary_report = f"""Job Verification Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

JOB VERIFICATION:
Classification: {results['job_verification']['label']}
Analysis: {results['job_verification']['reasoning']}

COMPANY RESEARCH:
{f"Company: {company_data['name']}" if results.get('company_research') else "No company information available"}
{company_data['summary'] if results.get('company_research') and company_data['summary'] else ""}
"""
                            st.download_button(
                                "📝 Download Summary",
                                data=summary_report,
                                file_name=f"job_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                        
                        # Save to history
                        save_results_to_session('comprehensive', results)
                        
                    except Exception as e:
                        st.error(f"❌ Error during comprehensive analysis: {str(e)}")
    
    # Tab 4: History and analytics
    
    with tab4:
        st.markdown('<h2 class="sub-header">Analysis History & Analytics</h2>', unsafe_allow_html=True)
        
        if 'verification_history' not in st.session_state or not st.session_state.verification_history:
            st.info("📊 No analysis history available yet. Start by verifying some job posts!")
            return
        
        # Summary statistics
        st.markdown("### 📈 Summary Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        total_analyses = len(st.session_state.verification_history)
        job_verifications = len([e for e in st.session_state.verification_history if e['type'] == 'job_verification'])
        company_researches = len([e for e in st.session_state.verification_history if e['type'] == 'company_research'])
        comprehensive_analyses = len([e for e in st.session_state.verification_history if e['type'] == 'comprehensive'])
        
        with col1:
            st.metric("Total Analyses", total_analyses)
        with col2:
            st.metric("Job Verifications", job_verifications)
        with col3:
            st.metric("Company Researches", company_researches)
        with col4:
            st.metric("Comprehensive", comprehensive_analyses)
        
        # Visualization
        chart = create_verification_chart()
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        
        # Detailed history
        st.markdown("### 📋 Detailed History")
        
        # History table
        history_data = []
        for entry in reversed(st.session_state.verification_history[-20:]):  # Show last 20
            if entry['type'] == 'job_verification':
                history_data.append({
                    'Timestamp': entry['timestamp'],
                    'Type': 'Job Verification',
                    'Result': entry['data']['job_verification']['label'],
                    'Preview': entry['data'].get('job_text_preview', 'N/A')[:100]
                })
            elif entry['type'] == 'company_research':
                history_data.append({
                    'Timestamp': entry['timestamp'],
                    'Type': 'Company Research',
                    'Result': entry['data']['company_info']['name'],
                    'Preview': entry['data']['company_info'].get('summary', 'N/A')[:100]
                })
            elif entry['type'] == 'comprehensive':
                history_data.append({
                    'Timestamp': entry['timestamp'],
                    'Type': 'Comprehensive',
                    'Result': entry['data']['job_verification']['label'],
                    'Preview': f"Job: {entry['data']['job_verification']['label']}"
                })
        
        if history_data:
            df = pd.DataFrame(history_data)
            st.dataframe(df, use_container_width=True)
            
            # Export history
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download History as CSV",
                data=csv,
                file_name=f"verification_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        # Clear history option
        if st.button("🗑️ Clear History", type="secondary"):
            if st.button("⚠️ Confirm Clear History"):
                st.session_state.verification_history = []
                st.success("✅ History cleared!")
                st.rerun()

if __name__ == "__main__":
    main()