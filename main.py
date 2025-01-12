import json
import os
import logging
import taipy.gui.builder as tgb
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, SystemMessage
from langchain.memory import ConversationBufferMemory
from taipy.gui import Gui, notify
from wordcloud import WordCloud
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('agg')
import io
import base64

load_dotenv()

query_message = ""
messages = []
messages_dict = {}
path_upload = ""
resume_text = ""
metrics_data = {}
skills_list = []
roles_list = []
companies_list = []
show_analysis = False
is_uploading = False
upload_progress = 0
upload_status = "Ready"
skills_wordcloud = None
skills_chart_data = dict()
roles_chart_data = dict()
companies_chart_data = dict()
skills_chart_layout = {
    "title": "Skills Distribution (Soft vs Tech)"
}
roles_chart_layout = {
    "title": "Roles Distribution (Managerial vs Individual Contributor)"
}
companies_chart_layout = {
    "title": "Companies Distribution (Years of Working)"
}
improve_suggestion = ""
conversation_chain = None
memory = ConversationBufferMemory(return_messages=True)
chat_history = []
active_cls = "fullwidth nav-button active"
inactive_cls = "fullwidth nav-button"
insight_cls = inactive_cls
chat_cls = inactive_cls
grammar_cls = inactive_cls

logging.basicConfig(level=logging.INFO)

system_prompt = """You are a helpful assistant specialized in analyzing resumes. 
Follow these rules:
1. Provide detailed analysis of resume content
2. Highlight key skills and experiences
3. Suggest improvements when asked
4. Be professional and constructive"""

current_page = "insights"
def switch_page(state, page):
    state.current_page = page
    if page == "insights":
        state.insight_cls = active_cls
        state.chat_cls = inactive_cls
        state.grammar_cls = inactive_cls
    elif page == "chat":
        state.insight_cls = inactive_cls
        state.chat_cls = active_cls
        state.grammar_cls = inactive_cls
    elif page == "grammar":
        state.insight_cls = inactive_cls
        state.chat_cls = inactive_cls
        state.grammar_cls = active_cls

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    top_p=0.3,
    top_k=30,
    max_output_tokens=4096,
    system_prompt=system_prompt
)

def on_init(state):
    state.messages = [
        {
            "style": "assistant_message",
            "content": "Hi, I am Resume AI assistant! How can I help you today?",
        },
    ]
    new_conv = create_conv(state)
    state.conv.update_content(state, new_conv)

def create_conv(state):
    messages_dict = {}
    with tgb.Page() as conversation:
        for i, message in enumerate(state.messages):
            text = message["content"].replace("<br>", "\n").replace('"', "'")
            messages_dict[f"message_{i}"] = text
            tgb.text(
                f"{{messages_dict.get('message_{i}') or ''}}",
                class_name=f"message_base {message['style']}",
                mode="md",
            )
    state.messages_dict = messages_dict
    return conversation


def send_message(state):
    if not state.conversation_chain:
        state.conversation_chain = initialize_chat(state.resume_text)

    state.messages.append(
        {
            "style": "user_message",
            "content": state.query_message,
        }
    )
    
    state.conv.update_content(state, create_conv(state))
    notify(state, "info", "Sending message...")
    state.messages.append(
        {
            "style": "assistant_message",
            "content": state.conversation_chain.predict(input=state.query_message),
        }
    )
    state.conv.update_content(state, create_conv(state))
    state.query_message = ""


def reset_chat(state):
    state.query_message = ""
    on_init(state)

def reset_state(state):
    state.resume_text = ""
    state.skills_list = []
    state.roles_list = []
    state.companies_list = []
    state.metrics_data = {}
    state.skills_wordcloud = None
    state.skills_chart_data = dict()
    state.roles_chart_data = dict()
    state.companies_chart_data = dict()
    state.improve_suggestion = ""
    state.show_analysis = False
    state.is_uploading = False
    state.upload_progress = 0
    state.upload_status = "Ready"
    state.skills_wordcloud = None
    state.improve_suggestion = ""
    state.conversation_chain = None
    state.chat_history = []
    state.current_page = "insights"

def generate_wordcloud(skills):
    # Create WordCloud
    try:
        # Create and configure WordCloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            prefer_horizontal=0.7
        ).generate(' '.join(skills))
        
        # Save to byte stream
        buf = io.BytesIO()
        wordcloud.to_image().save(buf, 'png')
        buf.seek(0)
        
        # Encode to base64
        img_str = buf.getvalue()
        buf.close()
        
        return img_str
        
    except Exception as e:
        logging.error(f"WordCloud generation failed: {e}")
        return ""

def handle_upload(state):
    try:
        # Reset
        reset_state(state)
        reset_chat(state)
        switch_page(state, "insights")

        # Set initial upload state
        state.is_uploading = True
        state.upload_progress = 10
        state.upload_status = "Loading PDF..."

        # Load PDF content using PyMuPDFLoader
        loader = PyMuPDFLoader(state.path_upload)
        state.upload_progress = 30
        state.upload_status = "Processing content..."
        
        documents = loader.load()
        state.upload_progress = 50
        state.upload_status = "Analyzing resume..."

        state.resume_text = ""
        for doc in documents:
            state.resume_text += f"{doc.page_content}"

        state.upload_progress = 70
        state.upload_status = "Generating insights..."
        
        context_prompt = f"""Resume Content:
        {state.resume_text}
        
        Analyze the resume and return a JSON object with the following information:
        - age: estimated age based on experience (if available)
        - gender: inferred gender (if available)
        - num_skills: total number of technical and soft skills (0 if none)
        - num_soft_skills: number of soft skills (0 if none)
        - num_technical_skills: number of technical skills (0 if none)
        - skills: list of technical and soft skills
        - years_experience: total years of professional experience
        - num_roles: total number of job positions held (0 if none)
        - roles: list of job positions held, extract all, don't leave single one behind
        - num_managerial_roles: number of managerial roles held (0 if none), only consider role as managerial if there's Lead, Manager, VP, Director, C-level indicated in the job title
        - num_individual_contributor_roles: number of individual contributor roles held (0 if none)
        - num_companies: total number of companies worked for (0 if none)
        - companies: list of companies worked for along with years of working (use fractions not only round number), extract all don't leave single one behind
        put N/A if the information is not available.
        
        Format your response as a valid JSON object only. Example:
        {{
            "age": 30,
            "gender": "male",
            "num_skills": 3,
            "num_soft_skills": 1,
            "num_technical_skills": 2,
            "skills": ["Python", "Machine Learning", "Data Analysis"],
            "years_experience": 8,
            "num_roles": 4,
            "roles": ["Data Scientist", "Machine Learning Engineer", "Data Analyst", "Research Scientist"],
            "num_managerial_roles": 1,
            "num_individual_contributor_roles": 3,
            "num_companies": 3,
            "companies": {{
                "Company A": 3,
                "Company B": 2,
                "Company C": 1
            }}
        }}
        Think step by step before answering.
        """    
        # Get response using content-aware prompt
        response = llm.invoke(context_prompt).content
        response = response.replace("```json", '').replace("```", '')

        improve_prompt = f"""Resume Content:
        {state.resume_text}
        
        Analyze the resume and check if there's a grammar or spelling mistake mistake (list the sentence with grammar mistake) and potential improvement in the resume.
        Improvement could be in the form of a better sentence structure, more concise language, or a more effective use of keywords.
        It can also be suggestion to add more skills, experience, or achievements.
        Suggest job positions to apply based on the resume.
        
        Format your response as a markdown with 2 headers Grammar Check and Potential Improvement. Example:
        ## Grammar Check
        - List of grammar mistakes in detail with what sentence in wrong grammar
        ## Potential Improvement
        - List of potential improvements
        ## Suggested Job Positions
        - List of suggested job positions

        Think step by step before answering and provide a detailed explanation for each improvement.
        """    
        # Get response using content-aware prompt
        improve_response = llm.invoke(improve_prompt).content
        state.improve_suggestion = improve_response

        # Parse JSON response
        analysis = json.loads(response)
        
        # Create visualization data
        state.metrics_data = {
            "labels": ["Skills", "Years Experience", "Roles", "Companies", "Soft Skills", "Technical Skills", "Managerial Roles", "Individual Contributor Roles"],
            "values": [
                analysis["num_skills"],
                analysis["years_experience"],
                analysis["num_roles"],
                analysis["num_companies"],
                analysis["num_soft_skills"],
                analysis["num_technical_skills"],
                analysis["num_managerial_roles"],
                analysis["num_individual_contributor_roles"]
            ]
        }
        
        # Store lists for detailed view
        state.skills_wordcloud = generate_wordcloud(analysis['skills'])
        
        state.roles_list = analysis["roles"]
        state.companies_list = analysis["companies"]

        state.upload_progress = 100
        state.upload_status = "Complete!"
        state.is_uploading = False
        state.show_analysis = True

        state.skills_chart_data = {
            "values": [state.metrics_data["values"][4], state.metrics_data["values"][5]],
            "labels": [state.metrics_data["labels"][4], state.metrics_data["labels"][5]]
        }

        state.roles_chart_data = {
            "values": [state.metrics_data["values"][6], state.metrics_data["values"][7]],
            "labels": [state.metrics_data["labels"][6], state.metrics_data["labels"][7]]
        }

        state.companies_chart_data = {
            "years": state.companies_list.values(),
            "company": state.companies_list.keys(),
        }
        
        notify(state, "success", "Resume analysis complete!")
    except Exception as e:
        state.upload_status = f"Error: {str(e)}"
        state.is_uploading = False
        state.show_analysis = False
        notify(state, "error", f"Error processing resume: {str(e)}")

# Initialize chat model with resume context
def initialize_chat(resume_text):
    system_template = f"""You are an AI assistant specialized in analyzing resumes. 
    Use this resume content for context: {resume_text}
    Answer questions based on the resume content and maintain professional tone.
    Use your own expertise to provide insights and suggestions."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.7,
        max_output_tokens=4096,
        system_prompt=system_prompt
    )
    
    return ConversationChain(
        prompt=prompt,
        memory=memory,
        verbose=True,
        llm=llm
    )    

with tgb.Page() as page:
    with tgb.layout(columns="350px 1"):
        with tgb.part(class_name="sidebar"):
            tgb.text("## Resume AI", mode="md")
            
            tgb.file_selector(
                "{path_upload}",
                label="Upload Document",
                on_action=handle_upload,
                multiple=False, extensions=".pdf",
                class_name="fullwidth"
            )

            with tgb.part(render="{show_analysis}"):                
                tgb.button(
                    "Resume Insights", 
                    on_action=lambda s: switch_page(s, "insights"),
                    class_name="{insight_cls}"
                )
                tgb.button(
                    "Grammar Check and Suggestions", 
                    on_action=lambda s: switch_page(s, "grammar"),
                    class_name="{grammar_cls}"
                )
                tgb.button(
                    "Chat with AI", 
                    on_action=lambda s: switch_page(s, "chat"),
                    class_name="{chat_cls}"
                )

        with tgb.part(class_name="p1"):
            with tgb.part(render="{is_uploading}"):
                tgb.progress("{upload_progress}")
                tgb.text("{upload_status}", class_name="upload-status")

            with tgb.part("Analysis Results", class_name="card", render="{show_analysis and current_page == 'insights'}"):
                # Bar chart for metrics
                with tgb.layout(columns="1fr 1fr 1fr 1fr"):
                    # Skills Metric
                    with tgb.part(class_name="metric-card"):
                        tgb.text("## 💡", mode="md")
                        tgb.text("### Number of Skills", mode="md")
                        tgb.text("**{metrics_data['values'][0]}**", mode="md", class_name="metric-value")

                    # Experience Metric
                    with tgb.part(class_name="metric-card"):
                        tgb.text("## 📈", mode="md")
                        tgb.text("### Years of Experience", mode="md")
                        tgb.text("**{metrics_data['values'][1]}**", mode="md", class_name="metric-value")

                    # Roles Metric
                    with tgb.part(class_name="metric-card"):
                        tgb.text("## 👔", mode="md")
                        tgb.text("### Number of Roles", mode="md")
                        tgb.text("**{metrics_data['values'][2]}**", mode="md", class_name="metric-value")

                    # Companies Metric
                    with tgb.part(class_name="metric-card"):
                        tgb.text("## 🏢", mode="md")
                        tgb.text("### Number of Companies", mode="md")
                        tgb.text("**{metrics_data['values'][3]}**", mode="md", class_name="metric-value")
                
                # Skills section
                with tgb.part("Skills Overview", class_name="mt-3", render="{show_analysis}"):
                    tgb.text("### Skills Overview", mode="md")
                    with tgb.layout(columns="1 1"):
                        with tgb.part("Skills Word Cloud", class_name="centered-container"):
                            tgb.image(content="{skills_wordcloud}", width="100%", class_name="centered-image")
                        with tgb.part("Skills Classification", class_name="centered-container"):    
                            tgb.chart("{skills_chart_data}", type="pie", values="values", labels="labels", rebuild=True, layout="{skills_chart_layout}")
                
                # Experience section
                with tgb.part("Work Experience", class_name="mt-3"):
                    tgb.text("### Work Experience Overview", mode="md")
                    with tgb.layout(columns="1 1"):
                        with tgb.part("Work Experience Job Roles", class_name="centered-container"):
                            tgb.chart("{roles_chart_data}", type="pie", values="values", labels="labels", rebuild=True, layout="{roles_chart_layout}")
                        with tgb.part("Work Experience Companies", class_name="centered-container"):    
                            tgb.chart("{companies_chart_data}", type="bar", y="years", x="company", rebuild=True, layout="{companies_chart_layout}")

            with tgb.part(render="{current_page == 'grammar' and show_analysis}"):
                tgb.text("{improve_suggestion}", mode="md")

            with tgb.part("Chat with AI", class_name="card", render="{current_page == 'chat' and show_analysis}"):    
                tgb.part(partial="{conv}", height="600px", class_name="card card_chat")
                with tgb.part("card mt1"):
                    tgb.input(
                        "{query_message}",
                        on_action=send_message,
                        change_delay=-1,
                        label="Write your message:",
                        class_name="fullwidth",
                    )


if __name__ == "__main__":
    gui = Gui(page)
    conv = gui.add_partial("")
    gui.run(title="Resume AI", host="0.0.0.0" dark_mode=False, margin="0px", debug=False, port=10000)
