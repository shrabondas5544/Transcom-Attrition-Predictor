import os
from django.conf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Thread-safe in-memory vector store cache
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store
        
    policies_path = os.path.join(settings.BASE_DIR, 'chatbot', 'data', 'transcom_retention_policies.txt')
    if not os.path.exists(policies_path):
        raise FileNotFoundError(f"Retention policies document not found at {policies_path}")
        
    with open(policies_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    
    # Initialize embeddings (using exact class name supported by installed package)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GEMINI_API_KEY
    )
    
    # Load into in-memory FAISS database
    _vector_store = FAISS.from_texts(docs, embeddings)
    return _vector_store

def _get_fallback_chatbot_response(user_query, error_message=None):
    """
    Generate a policy-compliant chatbot answer locally when the Gemini API is offline/quota exhausted.
    """
    import logging
    logger = logging.getLogger(__name__)
    if error_message:
        logger.warning(f"Chatbot Gemini API failure: {error_message}. Using local rule-based fallback response engine.")
        
    query_lower = user_query.lower()
    
    # Check for overtime
    has_overtime = "overtime" in query_lower or "fatigue" in query_lower or "hours" in query_lower
    # Check for commute/distance
    has_commute = "distance" in query_lower or "commute" in query_lower or "km" in query_lower or "location" in query_lower
    # Check for manager friction/engagement
    has_engagement = "engagement" in query_lower or "effectiveness" in query_lower or "friction" in query_lower or "manager" in query_lower
    # Check for salary
    has_salary = "salary" in query_lower or "compensation" in query_lower or "earnings" in query_lower
    
    response_parts = []
    
    if has_overtime:
        response_parts.append(
            "### Overtime Fatigue Countermeasures (Section 1)\n"
            "- **Weekly limit:** Enforce a maximum of 48 total working hours per week.\n"
            "- **Rest periods:** Maintain a minimum of 11 consecutive hours of daily rest between shifts, and no more than 6 consecutive working days without a mandatory 24-hour break.\n"
            "- **Overtime cap:** Strictly cap monthly overtime hours at 50 hours (recommended under 40 hours). Give 1 paid day of compensatory leave (Comp-Time) for every 8 hours of overtime worked above the 40-hour threshold, to be used within 30 days."
        )
    if has_commute:
        response_parts.append(
            "### Commute & Distance Mitigation (Section 2)\n"
            "- **Travel Stipend:** RESIDENCE exceeding 20 km from their designated outlet are eligible for a monthly travel stipend of 3,500 BDT.\n"
            "- **Regional Reassignments:** Field officers residing far from their assigned retail outlet can request a lateral transfer to the nearest hub, which regional HR must process within 14 business days.\n"
            "- **Hybrid Roster:** Non-outlet support teams are permitted a partial hybrid schedule of up to 2 remote work days per week, subject to approval."
        )
    if has_engagement:
        response_parts.append(
            "### Engagement & Manager Friction (Section 3)\n"
            "- **Stay Interviews:** Conduct quarterly Stay Interviews focusing on career path mapping rather than active tasks.\n"
            "- **Feedback loops:** Set up bi-weekly 30-minute 1-on-1 alignment sessions.\n"
            "- **Conflict Resolution:** Follow the 3-step path (Informal direct alignment -> HR-mediated communication review -> Lateral outlet transfer).\n"
            "- **Leadership Coaching:** Mandate managers with team effectiveness scores under 5/10 to attend the 'Empathetic Leadership at Transcom' coaching framework."
        )
    if has_salary:
        response_parts.append(
            "### Compensation Audit Guidelines\n"
            "- **Salary Review:** Perform a market pay alignment check to address potential salary discrepancies.\n"
            "- **Incentive Audit:** Review incentive and incentive earnings payout structures to make sure performance targets align clearly with bonus metrics."
        )
        
    if not response_parts:
        response_parts.append(
            "### Transcom Electronics Limited - Strategic Retention Guidelines\n"
            "- **Overtime Cap:** Ensure overtime hours do not exceed 40 hours/month and daily rest is at least 11 hours.\n"
            "- **Travel Support:** Residing >20km from the outlet qualifies for a 3,500 BDT commute stipend or lateral reassignment.\n"
            "- **Interpersonal Alignment:** Conduct quarterly Stay Interviews and bi-weekly feedback loops for at-risk staff."
        )
        
    plan_text = "\n\n".join(response_parts)
    
    title = "✨ **Transcom Chatbot Advisor (Policy-Based Fallback Response)**"
    disclaimer = ""
    if error_message:
        if "quota" in error_message.lower() or "429" in error_message or "exhausted" in error_message.lower():
            disclaimer = (
                "*Notice: Gemini API quota limit reached. Showing local, highly tailored policy-based fallback response.*"
            )
        elif "not configured" in error_message.lower() or "placeholder" in error_message.lower():
            disclaimer = (
                "System Notice: The GEMINI_API_KEY is currently not configured or using the default placeholder. Showing local, highly tailored policy-based fallback response."
            )

        else:
            disclaimer = (
                f"*Notice: Gemini API error ({error_message[:40]}...). Showing local, highly tailored policy-based fallback response.*"
            )
    else:
        disclaimer = (
            "*Notice: Local, highly tailored policy-based response.*"
        )
        
    return f"{title}\n{disclaimer}\n\n{plan_text}"

def generate_retention_response(user_query):
    """
    RAG pipeline: similarity search on FAISS + ChatGoogleGenerativeAI response generation.
    If the LLM call fails due to quota limit, missing API key, or networking issues,
    gracefully fall back to local rule-based policy engine.
    """
    # Defensive check for missing API Key
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == 'your_gemini_api_key_here' or settings.GEMINI_API_KEY == '':
        error_msg = "GEMINI_API_KEY is not configured or using default placeholder."
        return _get_fallback_chatbot_response(user_query, error_message=error_msg)
        
    try:
        # Get or build vector store
        db = get_vector_store()
        
        # Retrieve top 3 blocks
        docs = db.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3
        )
        
        # Prompt construction
        template = "You are an expert HR Advisor for Transcom Electronics Limited. Use the following policy context to give a precise, concise, and actionable retention strategy answer to the manager's problem. Context: {context} \n Question: {question}"
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
        prompt = prompt_template.format(context=context, question=user_query)
        
        # Invoke LLM
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return _get_fallback_chatbot_response(user_query, error_message=str(e))


def _get_fallback_prescription(employee_data, top_drivers, error_message=None):
    """
    Generate a beautiful, policy-compliant 3-step retention action plan locally
    when the Gemini API is rate-limited, quota-exhausted, or unconfigured.
    """
    import logging
    logger = logging.getLogger(__name__)
    if error_message:
        logger.warning(f"Gemini API failure: {error_message}. Using local rule-based fallback policy engine.")
    
    # Identify primary risk dimensions based on drivers and raw values
    drivers_lower = [d.get('feature', '').lower() for d in top_drivers] if top_drivers else []
    
    overtime = employee_data.get('overtime_hours', 0) or 0
    distance = employee_data.get('distance_from_workplace', 0) or 0
    salary = employee_data.get('monthly_salary', 0) or 0
    mgr_effectiveness = employee_data.get('manager_effectiveness_score', 10) or 10
    engagement = employee_data.get('employee_engagement_score', 10) or 10
    location = employee_data.get('location', 'Unknown')
    
    has_overtime_issue = 'overtime hours' in drivers_lower or overtime > 40
    has_commute_issue = 'distance from workplace' in drivers_lower or distance > 20
    has_salary_issue = 'monthly salary' in drivers_lower or salary < 22000
    has_mgr_issue = 'manager effectiveness score' in drivers_lower or mgr_effectiveness < 4
    has_engagement_issue = 'employee engagement score' in drivers_lower or engagement < 4
    
    steps = []
    
    # Action 1: Overtime fatigue mitigation
    if has_overtime_issue:
        steps.append(
            f"**1. Shift Rota Cap & Rest Enforcement (Overtime: {overtime} hrs/mo):** "
            "Strictly enforce Transcom's maximum shift cap (no more than 48 working hours/week) "
            "and guarantee at least 11 consecutive hours of daily rest. Limit monthly overtime to under 40 hours. "
            "Grant 1 paid day of compensatory leave (Comp-Time) for every 8 hours of overtime worked above the 40-hour threshold."
        )
    
    # Action 2: Distance/Commute mitigation
    if has_commute_issue:
        steps.append(
            f"**2. Commute Stipend & Regional Hub Reassignment (Distance: {distance} km):** "
            f"Residing {distance} km from work exceeds the target threshold. Approve a monthly travel stipend of 3,500 BDT "
            f"or register them for shared shuttle routing. Direct regional HR to process a lateral transfer request "
            f"to the nearest outlet or service hub in {location} within 14 business days."
        )
        
    # Action 3: Salary / Incentive alignment
    if has_salary_issue:
        steps.append(
            f"**3. Compensation Realignment & Incentive Evaluation (Salary: {salary:,} BDT):** "
            "Conduct a structured salary review comparing the employee's current pay against market baselines. "
            "Audit performance metrics against incentive payouts to clarify bonus opportunities, "
            "and establish clear performance-linked objectives to elevate their monthly earnings potential."
        )
        
    # Action 4: Manager Effectiveness / Stay Interview
    if has_mgr_issue or has_engagement_issue:
        steps.append(
            f"**4. Empathetic Alignment & Stay Interview (Mgr Score: {mgr_effectiveness}/10, Engagement: {engagement}/10):** "
            "Conduct a structured stay interview within 7 days focusing on long-term career mapping rather than active tasks. "
            "If manager friction persists, initiate a mediated HR review or facilitate a lateral reassignment to a different outlet. "
            "Mandate the supervisor's participation in Transcom's Empathetic Leadership training."
        )
        
    # Standard baseline fallback plans to ensure we have exactly 3 highly specific steps
    if len(steps) < 1:
        steps.append(
            "**1. Quarterly Stay Interview & Career Mapping:** Conduct a structured stay interview focusing on "
            "long-term professional goals, career development paths within Transcom, and general job satisfaction."
        )
    if len(steps) < 2:
        steps.append(
            "**2. Workload & Roster Review:** Perform a proactive roster review to ensure workload is balanced, "
            "ensuring the employee has adequate rest intervals and is not experiencing unflagged role fatigue."
        )
    if len(steps) < 3:
        steps.append(
            "**3. Recognition & Professional Development:** Audit the employee's training log and performance "
            "recognition. Map targeted training milestones to promote retail advancement opportunities."
        )
        
    selected_steps = steps[:3]
    plan_text = "\n\n".join(selected_steps)
    
    title = "✨ **Transcom HR Advisor (Policy-Based Retention Plan)**"
    disclaimer = ""
    if error_message:
        if "quota" in error_message.lower() or "429" in error_message or "exhausted" in error_message.lower():
            disclaimer = (
                "*Notice: Gemini API quota limit reached. Showing local, highly tailored policy-based fallback plan.*"
            )
        elif "not configured" in error_message.lower() or "placeholder" in error_message.lower():
            disclaimer = (
                "*Notice: GEMINI_API_KEY not configured. Showing local, highly tailored policy-based fallback plan.*"
            )
        else:
            disclaimer = (
                f"*Notice: Gemini API error ({error_message[:40]}...). Showing local, highly tailored policy-based fallback plan.*"
            )
    else:
        disclaimer = (
            "*Notice: Local, highly tailored policy-based retention plan.*"
        )
        
    return f"{title}\n{disclaimer}\n\n{plan_text}"

def generate_individual_prescription(employee_data, top_drivers):
    """
    RAG pipeline: similarity search on FAISS using top drivers + ChatGoogleGenerativeAI to
    produce a highly tailored 3-step action plan to retain a specific individual.
    If the LLM call fails due to quota limit, missing API key, or networking issues,
    gracefully fall back to local rule-based policy engine.
    """
    # Defensive check for missing API Key
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == 'your_gemini_api_key_here' or settings.GEMINI_API_KEY == '':
        error_msg = "GEMINI_API_KEY is not configured or using default placeholder."
        return _get_fallback_prescription(employee_data, top_drivers, error_message=error_msg)
        
    try:
        # Build query based on top drivers to fetch relevant policy blocks
        query = ", ".join([d['feature'] for d in top_drivers]) if top_drivers else "retention"
        
        # Get or build vector store
        db = get_vector_store()
        
        # Retrieve top 3 blocks
        docs = db.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3
        )
        
        # Format drivers display string
        drivers_str = ", ".join([f"{d['feature']} (SHAP: {d['shap_value']:.4f})" for d in top_drivers]) if top_drivers else "None (Stable)"
        prob_str = f"{round(employee_data['attrition_probability'] * 100, 1)}%" if employee_data['attrition_probability'] is not None else "N/A"
        
        # Prompt construction matching user prompt exactly
        template = (
            "You are an elite corporate HR strategist for Transcom Electronics Limited. "
            "Analyze this specific Field Officer profile: "
            "Age: {age}, Gender: {gender}, Location: {location}, Monthly Salary: {salary} BDT, "
            "Overtime: {overtime} hours, Distance: {distance} km. Their calculated flight risk is {prob} "
            "and their primary breakdown drivers are {drivers}. "
            "Based strictly on the provided company retention policies context, generate a bulleted, "
            "highly tailored 3-step action plan to retain this specific individual. Do not use generic advice. "
            "Context: {context}"
        )
        
        prompt_template = PromptTemplate(
            template=template, 
            input_variables=["age", "gender", "location", "salary", "overtime", "distance", "prob", "drivers", "context"]
        )
        prompt = prompt_template.format(
            age=employee_data['age'],
            gender=employee_data['gender'],
            location=employee_data['location'],
            salary=employee_data['monthly_salary'],
            overtime=employee_data['overtime_hours'],
            distance=employee_data['distance_from_workplace'],
            prob=prob_str,
            drivers=drivers_str,
            context=context
        )
        
        # Invoke LLM
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return _get_fallback_prescription(employee_data, top_drivers, error_message=str(e))


