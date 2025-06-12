import os
import logging
import requests
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema import Document
import json
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class JobPostResult:
    """Result from job post verification"""
    reasoning: str
    label: str
    confidence: Optional[float] = None
    raw_response: str = ""

@dataclass
class CompanyInfo:
    """Structured company information"""
    name: str
    founding_date: Optional[str] = None
    headquarters: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    website_link: Optional[str] = None
    summary: Optional[str] = None

@dataclass
class SearchResult:
    """Generic search result structure"""
    title: str
    content: str
    url: str
    source: str

# ============================================================================
# SEARCH PROVIDERS
# ============================================================================

class SearchAPIProvider:
    """SearchAPI provider for web search"""
    
    def __init__(self, api_key: str, engine: str = "google"):
        self.api_key = api_key
        self.engine = engine
        self.base_url = "https://www.searchapi.io/api/v1/search"
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform search using SearchAPI"""
        try:
            params = {
                'engine': self.engine,
                'q': query,
                'api_key': self.api_key,
                'num': num_results
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            search_results = response.json()
            results = []
            
            for result in search_results.get('organic_results', []):
                results.append(SearchResult(
                    title=result.get('title', ''),
                    content=result.get('snippet', ''),
                    url=result.get('link', ''),
                    source='searchapi'
                ))
            
            logger.info(f"SearchAPI: Retrieved {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"SearchAPI search failed: {e}")
            return []

class GoogleCSEProvider:
    """Google Custom Search Engine provider"""
    
    def __init__(self, api_key: str, cse_id: str):
        self.api_key = api_key
        self.cse_id = cse_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Perform search using Google CSE"""
        try:
            params = {
                "key": self.api_key,
                "cx": self.cse_id,
                "q": query,
                "num": num_results
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            items = response.json().get("items", [])
            results = []
            
            for item in items:
                # Try to get extended description from pagemap
                pagemap = item.get("pagemap", {})
                meta = pagemap.get("metatags", [{}])[0]
                content = (meta.get("og:description") or 
                          meta.get("twitter:description") or 
                          item.get("snippet", ""))
                
                results.append(SearchResult(
                    title=item.get('title', ''),
                    content=content,
                    url=item.get('link', ''),
                    source='google_cse'
                ))
            
            logger.info(f"Google CSE: Retrieved {len(results)} results for query: {query[:50]}...")
            return results
            
        except Exception as e:
            logger.error(f"Google CSE search failed: {e}")
            return []

# ============================================================================
# SEARCH MANAGER
# ============================================================================

class SearchManager:
    """Manages multiple search providers with fallback"""
    
    def __init__(self):
        self.providers = []
        self._setup_providers()
    
    def _setup_providers(self):
        """Initialize available search providers"""
        # Try SearchAPI first
        search_api_key = st.secrets['SEARCHAPI_KEY']
        if search_api_key:
            self.providers.append(SearchAPIProvider(search_api_key))
            logger.info("SearchAPI provider initialized")
        
        # Fallback to Google CSE
        google_api_key = st.secrets['GOOGLE_API_KEY']
        google_cse_id =st.secrets['GOOGLE_CSE_ID']
        if google_api_key and google_cse_id:
            self.providers.append(GoogleCSEProvider(google_api_key, google_cse_id))
            logger.info("Google CSE provider initialized")
        
        if not self.providers:
            logger.warning("No search providers available")
    
    def search(self, query: str, num_results: int = 10) -> List[SearchResult]:
        """Search using available providers with fallback"""
        for provider in self.providers:
            try:
                results = provider.search(query, num_results)
                if results:
                    return results
            except Exception as e:
                logger.warning(f"Provider {provider.__class__.__name__} failed: {e}")
                continue
        
        logger.error("All search providers failed")
        return self._get_fallback_results(query)
    
    def _get_fallback_results(self, query: str) -> List[SearchResult]:
        """Provide fallback results when all providers fail"""
        if "job" in query.lower() or "verification" in query.lower():
            return [
                SearchResult(
                    title="Real Job Post Example",
                    content="Legitimate job posting with detailed requirements, company contact info, and professional description.",
                    url="",
                    source="fallback"
                ),
                SearchResult(
                    title="Fake Job Post Example", 
                    content="Vague job description, unrealistic salary, no company details, requires upfront payment.",
                    url="",
                    source="fallback"
                )
            ]
        return []

# ============================================================================
# LLM MANAGER
# ============================================================================

class GroqLLMManager:
    """Manager for Groq LLM"""
    
    def __init__(self, model_name: str = "llama-3.3-70b-versatile", temperature: float = 0.1):
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self._setup_llm()
    
    def _setup_llm(self):
        """Initialize Groq LLM"""
        try:
            groq_api_key = st.secrets['GROQ_API_KEY']
            if not groq_api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")
            
            self.llm = ChatGroq(
                model_name=self.model_name,
                temperature=self.temperature,
                groq_api_key=groq_api_key,
                max_tokens=1024
            )
            logger.info(f"Groq LLM initialized: {self.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            self.llm = None
    
    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the Groq LLM"""
        if not self.llm:
            raise ValueError("Groq LLM not initialized")
        
        try:
            response = self.llm.invoke(prompt, **kwargs)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            logger.error(f"Groq LLM invocation failed: {e}")
            raise

# ============================================================================
# CORE SERVICES
# ============================================================================

class JobPostVerificationService:
    """Service for verifying job posts"""
    
    def __init__(self, search_manager: SearchManager, llm_manager: GroqLLMManager):
        self.search_manager = search_manager
        self.llm_manager = llm_manager
        self.prompt_template = self._create_prompt_template()
    
    def _create_prompt_template(self):
        """Create the prompt template for job verification"""
        system_prompt = """You are an expert job post verification system. Classify job postings as 'Real' or 'Fake' based on indicators.

Key indicators of FAKE job posts:
- Unrealistic salary promises (extremely high for minimal qualifications)
- Vague job descriptions with no specific responsibilities
- Poor grammar and spelling errors
- Requests for upfront payments or personal financial information
- No legitimate company contact information or vague contact details
- Too-good-to-be-true offers (work from home, no experience needed, high pay)
- Pressure to act quickly or urgency tactics
- Generic job titles with no company-specific details
- Requests for personal information like SSN upfront

Key indicators of REAL job posts:
- Specific job requirements and detailed responsibilities
- Realistic salary ranges for the industry and role
- Professional language and proper formatting
- Legitimate company information with verifiable details
- Clear application process through official channels
- Detailed job location and work arrangements
- Professional email addresses and contact information
- Industry-standard benefits and compensation structure

Analyze the job post carefully and provide your assessment.

Respond in this exact format:
Reasoning: <detailed explanation of your analysis>
Label: <Real or Fake>"""

        user_prompt = """Context examples from search:
{context}

Analyze this job post:
{text}"""

        return ChatPromptTemplate([
            ("system", system_prompt),
            ("user", user_prompt)
        ])
    
    def verify_job_post(self, job_text: str) -> JobPostResult:
        """Verify if a job post is real or fake"""
        try:
            if not job_text or not job_text.strip():
                raise ValueError("Job post text cannot be empty")
            
            # Search for relevant examples
            search_results = self.search_manager.search(
                f"job post verification fake real examples scam indicators", 
                num_results=5
            )
            
            # Format context
            context_parts = []
            for i, result in enumerate(search_results, 1):
                context_parts.append(f"Example {i}: {result.title}\n{result.content}")
            
            context = "\n\n".join(context_parts) if context_parts else "No additional context available."
            
            # Generate classification using Groq LLM
            chain = self.prompt_template | self.llm_manager.llm
            result = chain.invoke({"context": context, "text": job_text})
            
            return self._parse_verification_result(result)
            
        except Exception as e:
            logger.error(f"Job verification failed: {e}")
            return JobPostResult(
                reasoning=f"Error during verification: {str(e)}",
                label="Unknown",
                raw_response=str(e)
            )
    
    def _parse_verification_result(self, raw_result: Union[str, Dict, Any]) -> JobPostResult:
        """Parse the LLM response into structured format"""
        try:
            # Convert to string if needed
            if hasattr(raw_result, 'content'):
                raw_text = raw_result.content
            elif isinstance(raw_result, dict):
                raw_text = raw_result.get('text', str(raw_result))
            else:
                raw_text = str(raw_result)
            
            # Extract reasoning and label
            reasoning = ""
            label = "Unknown"
            
            lines = raw_text.split('\n')
            for line in lines:
                line = line.strip()
                if line.lower().startswith('reasoning:'):
                    reasoning = line[10:].strip()
                elif line.lower().startswith('label:'):
                    label = line[6:].strip()
            
            # Validate and normalize label
            if label.lower() not in ['real', 'fake']:
                if 'real' in raw_text.lower():
                    label = 'Real'
                elif 'fake' in raw_text.lower():
                    label = 'Fake'
                else:
                    label = 'Unknown'
            
            return JobPostResult(
                reasoning=reasoning or "No reasoning provided",
                label=label.capitalize(),
                raw_response=raw_text
            )
            
        except Exception as e:
            logger.error(f"Error parsing verification result: {e}")
            return JobPostResult(
                reasoning=f"Error parsing response: {str(e)}",
                label="Unknown", 
                raw_response=str(raw_result)
            )

class CompanyResearchService:
    """Service for researching company information"""
    
    def __init__(self, search_manager: SearchManager, llm_manager: GroqLLMManager):
        self.search_manager = search_manager
        self.llm_manager = llm_manager
        self.extraction_prompt = self._create_extraction_prompt()
        self.summary_prompt = self._create_summary_prompt()
    
    def _create_extraction_prompt(self):
        """Create prompt for extracting company information"""
        return PromptTemplate.from_template("""
You are an expert at extracting company metadata from search results.
The user searched for: {name}

From these search results:
{snippets}

Extract the following information about the company:
- founding_date: When was the company founded? (year only)
- headquarters: Where is the company headquartered? (city, state/country)
- industry: What industry/sector is the company in?
- size: How many employees does the company have? (provide range if exact number not available)
- website_link: What is the company's official website URL?

If information is not available or unclear, indicate that the field is unknown.
If the company name seems invalid or no reliable information is found, mention this.

Provide your findings in a clear, structured format with each field clearly labeled.
""")
    
    def _create_summary_prompt(self):
        """Create prompt for summarizing company information"""
        return PromptTemplate.from_template("""
Based on this company information:
{company_data}

Write a concise, professional summary about the company (approximately 100-150 words) that includes:
- Company name and their primary business/service
- Key details like founding date, location, and company size (if available)
- Industry and main business focus
- Any notable achievements or market position

Make it informative, professional, and suitable for a business context. Start with "About the company:"
""")
    
    def research_company(self, company_name: str) -> CompanyInfo:
        """Research comprehensive information about a company"""
        try:
            # Clean company name for search
            query = company_name if not company_name.startswith("http") else company_name.split("/")[2]
            
            # Search for company information
            search_results = self.search_manager.search(
                f'"{query}" company information founding date headquarters industry size website',
                num_results=8
            )
            
            if not search_results:
                return CompanyInfo(
                    name=company_name,
                    summary="No information found for this company."
                )
            
            # Format search results
            snippets = "\n".join([f"Source: {r.title}\nContent: {r.content}\n" for r in search_results])
            
            # Extract structured information
            extraction_chain = self.extraction_prompt | self.llm_manager.llm
            extracted_info = extraction_chain.invoke({
                "name": company_name,
                "snippets": snippets
            })
            
            # Generate summary
            summary_chain = self.summary_prompt | self.llm_manager.llm  
            summary = summary_chain.invoke({
                "company_data": extracted_info.content if hasattr(extracted_info, 'content') else str(extracted_info)
            })
            
            return self._parse_company_info(
                company_name, 
                extracted_info.content if hasattr(extracted_info, 'content') else str(extracted_info),
                summary.content if hasattr(summary, 'content') else str(summary)
            )
            
        except Exception as e:
            logger.error(f"Company research failed: {e}")
            return CompanyInfo(
                name=company_name,
                summary=f"Error researching company: {str(e)}"
            )
    
    def _parse_company_info(self, name: str, extracted_info: str, summary: str) -> CompanyInfo:
        """Parse extracted information into structured format"""
        try:
            info = CompanyInfo(name=name, summary=summary)
            
            # More robust parsing
            lines = extracted_info.split('\n')
            for line in lines:
                line_lower = line.lower().strip()
                
                if 'founding' in line_lower or 'founded' in line_lower:
                    # Extract year from line
                    import re
                    years = re.findall(r'\b(19|20)\d{2}\b', line)
                    if years:
                        info.founding_date = years[0]
                
                elif 'headquarters' in line_lower or 'headquartered' in line_lower:
                    # Extract location after colon or common patterns
                    if ':' in line:
                        location = line.split(':', 1)[1].strip()
                        if location and not location.lower().startswith('unknown'):
                            info.headquarters = location
                
                elif 'industry' in line_lower and ':' in line:
                    industry = line.split(':', 1)[1].strip()
                    if industry and not industry.lower().startswith('unknown'):
                        info.industry = industry
                
                elif ('size' in line_lower or 'employees' in line_lower) and ':' in line:
                    size = line.split(':', 1)[1].strip()
                    if size and not size.lower().startswith('unknown'):
                        info.size = size
                
                elif 'website' in line_lower:
                    import re
                    # Look for URLs in the line
                    urls = re.findall(r'https?://[^\s]+', line)
                    if urls:
                        info.website_link = urls[0]
                    elif ':' in line:
                        # Check if there's a domain after colon
                        potential_url = line.split(':', 1)[1].strip()
                        if '.' in potential_url and not potential_url.lower().startswith('unknown'):
                            info.website_link = potential_url if potential_url.startswith('http') else f"https://{potential_url}"
            
            return info
            
        except Exception as e:
            logger.error(f"Error parsing company info: {e}")
            return CompanyInfo(name=name, summary=summary)

# ============================================================================
# MAIN UNIFIED SERVICE
# ============================================================================

class UnifiedVerificationService:
    """Main service that combines job verification and company research"""
    
    def __init__(self):
        self.search_manager = SearchManager()
        self.groq_llm = GroqLLMManager()
        
        # Initialize services - now both use Groq LLM
        self.job_service = JobPostVerificationService(
            self.search_manager, 
            self.groq_llm
        )
        self.company_service = CompanyResearchService(
            self.search_manager,
            self.groq_llm
        )
    
    def verify_job_post(self, job_text: str) -> JobPostResult:
        """Verify a job post"""
        return self.job_service.verify_job_post(job_text)
    
    def research_company(self, company_name: str) -> CompanyInfo:
        """Research company information"""
        return self.company_service.research_company(company_name)
    
    def comprehensive_verification(self, job_text: str) -> Dict[str, Any]:
        """Perform comprehensive verification including company research"""
        try:
            # Verify job post
            job_result = self.verify_job_post(job_text)
            
            # Extract company name from job post
            company_name = self._extract_company_name(job_text)
            company_info = None
            
            if company_name:
                company_info = self.research_company(company_name)
            
            return {
                "job_verification": {
                    "label": job_result.label,
                    "reasoning": job_result.reasoning,
                    "confidence": job_result.confidence
                },
                "company_research": {
                    "name": company_info.name if company_info else None,
                    "summary": company_info.summary if company_info else None,
                    "details": {
                        "founding_date": company_info.founding_date if company_info else None,
                        "headquarters": company_info.headquarters if company_info else None,
                        "industry": company_info.industry if company_info else None,
                        "size": company_info.size if company_info else None,
                        "website": company_info.website_link if company_info else None
                    }
                } if company_info else None
            }
            
        except Exception as e:
            logger.error(f"Comprehensive verification failed: {e}")
            return {
                "job_verification": {
                    "label": "Unknown",
                    "reasoning": f"Error during verification: {str(e)}",
                    "confidence": None
                },
                "company_research": None
            }
    
    def _extract_company_name(self, job_text: str) -> Optional[str]:
        """Extract company name from job text using improved patterns"""
        try:
            lines = job_text.split('\n')
            
            # Look for common company name patterns
            for line in lines:
                line = line.strip()
                line_lower = line.lower()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Look for explicit company indicators
                company_indicators = [
                    'company:', 'company name:', 'employer:', 'organization:',
                    'corp', 'inc', 'ltd', 'llc', 'co.', 'corporation', 'limited'
                ]
                
                if any(indicator in line_lower for indicator in company_indicators):
                    # Clean up the line and extract company name
                    for indicator in ['company:', 'company name:', 'employer:', 'organization:']:
                        if indicator in line_lower:
                            company_name = line.split(':', 1)[1].strip()
                            if company_name:
                                return company_name
                    
                    # For corp, inc, etc., take the whole line if it's short enough
                    if len(line.split()) <= 5:
                        return line.strip()
            
            # Fallback: look for title-like capitalized text in first few lines
            for line in lines[:3]:
                line = line.strip()
                if line and len(line.split()) <= 4:
                    words = line.split()
                    # Check if most words are capitalized (likely company name)
                    capitalized_words = [w for w in words if w and w[0].isupper()]
                    if len(capitalized_words) >= len(words) * 0.7:  # 70% capitalized
                        return line
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting company name: {e}")
            return None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def setup_environment() -> bool:
    # """Check if required environment variables are set"""
    # required_vars = ['GROQ_API_KEY']
    # optional_vars = ['SEARCHAPI_KEY', 'GOOGLE_API_KEY', 'GOOGLE_CSE_ID']
    
    # missing_required = [var for var in required_vars if not st.secrets[var]]
    # if missing_required:
    #     logger.error(f"Missing required environment variables: {missing_required}")
    #     return False
    
    # missing_optional = [var for var in optional_vars if not st.secrets[var]]
    # if missing_optional:
    #     logger.warning(f"Missing optional environment variables: {missing_optional}")
    
    return True

def main():
    """Example usage of the unified system"""
    if not setup_environment():
        logger.error("Environment setup failed")
        return
    
    try:
        # Initialize the unified service
        service = UnifiedVerificationService()
        
        # Test job post verification
        test_job = """
        Software Developer - TechCorp Inc.
        
        We are seeking a skilled Software Developer for our San Francisco office.
        
        Responsibilities:
        - Develop web applications using Python and Django
        - Collaborate with cross-functional teams
        - Write clean, maintainable code
        
        Requirements:
        - Bachelor's degree in Computer Science
        - 3+ years experience in web development
        - Proficiency in Python, JavaScript, HTML/CSS
        
        Salary: $80,000 - $120,000
        Benefits: Health insurance, 401k, flexible PTO
        
        Contact: careers@techcorp.com
        Apply through our website: www.techcorp.com/careers
        """
        
        print("=== Job Post Verification ===")
        job_result = service.verify_job_post(test_job)
        print(f"Label: {job_result.label}")
        print(f"Reasoning: {job_result.reasoning}")
        print()
        
        # Test company research
        print("=== Company Research ===")
        company_result = service.research_company("Microsoft")
        print(f"Company: {company_result.name}")
        print(f"Summary: {company_result.summary}")
        if company_result.founding_date:
            print(f"Founded: {company_result.founding_date}")
        if company_result.headquarters:
            print(f"Headquarters: {company_result.headquarters}")
        if company_result.industry:
            print(f"Industry: {company_result.industry}")
        print()
        
        # Test comprehensive verification
        print("=== Comprehensive Verification ===")
        comprehensive_result = service.comprehensive_verification(test_job)
        print(json.dumps(comprehensive_result, indent=2))
        
    except Exception as e:
        logger.error(f"Error in main: {e}")

if __name__ == "__main__":
    print('done')
    # main()