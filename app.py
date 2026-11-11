import streamlit as st
from openai import OpenAI
import time
import base64
import uuid
import sqlite3
import datetime

st.set_page_config(page_title="Multi-Agent Hub", page_icon="⚡", layout="wide")

# --- SETUP CLIENT (Pointing to Cloudflare Worker) ---
client = OpenAI(
    base_url="https://nvidia-proxy.vellock.workers.dev/v1", 
    api_key="unused", 
    max_retries=0 
)

# --- DATABASE SETUP ---
@st.cache_resource
def get_db_connection():
    conn = sqlite3.connect("chat_history.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_id TEXT,
            role TEXT,
            content TEXT,
            image_base64 TEXT,
            FOREIGN KEY(chat_id) REFERENCES chats(id)
        )
    """)
    conn.commit()
    return conn

conn = get_db_connection()

# --- HELPER DATABASE FUNCTIONS ---
def create_new_chat():
    chat_id = str(uuid.uuid4())
    now = datetime.datetime.now()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO chats (id, title, created_at) VALUES (?, ?, ?)", (chat_id, "New Chat", now))
    conn.commit()
    return chat_id

def get_all_chats():
    cursor = conn.cursor()
    cursor.execute("SELECT id, title FROM chats ORDER BY created_at DESC")
    return cursor.fetchall()

def get_messages(chat_id):
    cursor = conn.cursor()
    cursor.execute("SELECT role, content, image_base64 FROM messages WHERE chat_id = ? ORDER BY id ASC", (chat_id,))
    rows = cursor.fetchall()
    messages = []
    for row in rows:
        role, content, img = row
        msg_content = [{"type": "text", "text": content}]
        if img:
            msg_content.append({"type": "image_url", "image_url": {"url": img}})
        messages.append({"role": role, "content": msg_content})
    return messages

def save_message(chat_id, role, text, image_base64=None):
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (chat_id, role, content, image_base64) VALUES (?, ?, ?, ?)", 
                   (chat_id, role, text, image_base64))
    conn.commit()

def update_chat_title(chat_id, new_title):
    cursor = conn.cursor()
    cursor.execute("UPDATE chats SET title = ? WHERE id = ?", (new_title, chat_id))
    conn.commit()

def delete_chat(chat_id):
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
    cursor.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
    conn.commit()

def stream_ai_response(system_prompt, message_history, model_name):
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            messages = [{"role": "system", "content": system_prompt}] + message_history
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True 
            )
            for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
            return
        except Exception as e:
            if attempt < max_attempts - 1:
                yield f"\n*(Connection lost. Quick-retrying attempt {attempt+1}/3...)*\n"
                time.sleep(2)
            else:
                yield f"\n\n**Final Error:** {e}"

# --- Model labels (The Cloudflare Worker handles the actual model routing) ---
models = {
    "Auto-Route (Fast)": "auto",
    "Auto-Route (Strong)": "auto"
}

# --- MAIN SIDEBAR HEADER ---
st.sidebar.header("App Settings")
mode = st.sidebar.radio("Choose your mode:", ("Normal Chat", "Pipeline", "Conversation", "Debate"))
st.sidebar.markdown("---")

# ==========================================
# MODE 1: NORMAL CHAT
# ==========================================
if mode == "Normal Chat":
    st.title("💬 Normal Chat")
    
    st.sidebar.subheader("Chat History")
    
    if st.sidebar.button("➕ New Chat", use_container_width=True):
        new_id = create_new_chat()
        st.session_state.current_chat_id = new_id
        st.rerun()

    past_chats = get_all_chats()
    
    if not past_chats:
        new_id = create_new_chat()
        st.session_state.current_chat_id = new_id
        st.rerun()
        
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = past_chats[0][0]

    for chat_id, title in past_chats:
        col1, col2 = st.sidebar.columns([0.8, 0.2])
        with col1:
            btn_type = "primary" if chat_id == st.session_state.current_chat_id else "secondary"
            if st.button(title, key=f"chat_{chat_id}", use_container_width=True, type=btn_type):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"del_{chat_id}", use_container_width=True):
                delete_chat(chat_id)
                if st.session_state.current_chat_id == chat_id:
                    del st.session_state.current_chat_id
                st.rerun()

    st.sidebar.markdown("---")
    
    system_prompt = st.sidebar.text_area("System Prompt:", "You are a helpful assistant.")
    selected_model_name = st.sidebar.selectbox("Select Model:", list(models.keys()))

    current_messages = get_messages(st.session_state.current_chat_id)

    for message in current_messages:
        with st.chat_message(message["role"]):
            if isinstance(message["content"], list):
                for item in message["content"]:
                    if item["type"] == "text":
                        st.markdown(item["text"])
                    elif item["type"] == "image_url":
                        st.image(item["image_url"]["url"])
            else:
                st.markdown(message["content"])

    if prompt := st.chat_input("Type your message here, or paste an image...", accept_file=True, file_type=["png", "jpg", "jpeg"]):
        
        user_text = prompt.text
        user_files = prompt.files
        
        image_base64_data = None
        if user_files:
            uploaded_file = user_files[0]
            base64_image = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
            image_base64_data = f"data:{uploaded_file.type};base64,{base64_image}"
            
            with st.chat_message("user"):
                st.image(uploaded_file)
                if user_text:
                    st.markdown(user_text)
        else:
            with st.chat_message("user"):
                st.markdown(user_text)

        save_message(st.session_state.current_chat_id, "user", user_text, image_base64_data)
        
        if current_messages == [] and user_text:
            new_title = user_text[:25] + "..." if len(user_text) > 25 else user_text
            update_chat_title(st.session_state.current_chat_id, new_title)

        with st.chat_message("assistant"):
            text_ph = st.empty()
            full_response = ""
            
            response_stream = stream_ai_response(system_prompt, current_messages + [{"role": "user", "content": [{"type": "text", "text": user_text if user_text else "Analyze this."}, {"type": "image_url", "image_url": {"url": image_base64_data}}] if image_base64_data else user_text}], models[selected_model_name])
            
            for chunk in response_stream:
                full_response += chunk
                text_ph.markdown(full_response)
            
            if "Final Error" in full_response:
                if st.button("Try Again"):
                    cursor = conn.cursor()
                    cursor.execute("DELETE FROM messages WHERE chat_id = ? AND role = 'user' ORDER BY id DESC LIMIT 1", (st.session_state.current_chat_id,))
                    conn.commit()
                    st.rerun()
            else:
                save_message(st.session_state.current_chat_id, "assistant", full_response)
                st.rerun()


# ==========================================
# MODE 2: PIPELINE
# ==========================================
elif mode == "Pipeline":
    st.title("🏗️ Pipeline")
    
    st.sidebar.subheader("Agent 1")
    p1_prompt = st.sidebar.text_area("Agent 1 Prompt", "Analyze logically.")
    m1 = st.sidebar.selectbox("Agent 1 Model", list(models.keys()), index=0)

    st.sidebar.subheader("Agent 2")
    p2_prompt = st.sidebar.text_area("Agent 2 Prompt", "Synthesize.")
    m2 = st.sidebar.selectbox("Agent 2 Model", list(models.keys()), index=1)

    question = st.text_input("Your Question:", "")
    
    if st.button("Run Pipeline"):
        if question:
            with st.chat_message("user"):
                st.markdown(question)
            
            with st.chat_message("assistant", avatar="📊"):
                text_ph1 = st.empty()
                ans1 = ""
                for chunk in stream_ai_response(p1_prompt, [{"role": "user", "content": question}], models[m1]):
                    ans1 += chunk
                    text_ph1.markdown(ans1)
            
            with st.chat_message("assistant", avatar="🏆"):
                text_ph2 = st.empty()
                ans2 = ""
                for chunk in stream_ai_response(p2_prompt, [{"role": "user", "content": f"Q: {question}\nDraft: {ans1}"}], models[m2]):
                    ans2 += chunk
                    text_ph2.markdown(ans2)

# ==========================================
# MODE 3: CONVERSATION
# ==========================================
elif mode == "Conversation":
    st.title("🗣️ Conversation")
    
    st.sidebar.subheader("Agent 1")
    p1_prompt = st.sidebar.text_area("Agent 1 Prompt", "You are an optimist.")
    m1 = st.sidebar.selectbox("Agent 1 Model", list(models.keys()), index=0)
    
    st.sidebar.subheader("Agent 2")
    p2_prompt = st.sidebar.text_area("Agent 2 Prompt", "You are a skeptic.")
    m2 = st.sidebar.selectbox("Agent 2 Model", list(models.keys()), index=1)
    
    rounds = st.sidebar.slider("Rounds:", 1, 3, 1)
    topic = st.text_input("Conversation Topic:", "")
    
    if st.button("Start Conversation"):
        if topic:
            with st.chat_message("user"):
                st.markdown(topic)
            history = f"Topic: {topic}\n\n"
            
            for i in range(1, rounds + 1):
                with st.chat_message("assistant", avatar="🟢"):
                    text_ph1 = st.empty()
                    ans1 = ""
                    for chunk in stream_ai_response(p1_prompt, [{"role": "user", "content": history}], models[m1]):
                        ans1 += chunk
                        text_ph1.markdown(ans1)
                    history += f"Agent 1: {ans1}\n\n"

                with st.chat_message("assistant", avatar="🔴"):
                    text_ph2 = st.empty()
                    ans2 = ""
                    for chunk in stream_ai_response(p2_prompt, [{"role": "user", "content": history}], models[m2]):
                        ans2 += chunk
                        text_ph2.markdown(ans2)
                    history += f"Agent 2: {ans2}\n\n"

# ==========================================
# MODE 4: DEBATE
# ==========================================
elif mode == "Debate":
    st.title("⚔️ Debate")
    
    st.sidebar.subheader("Pro Side")
    pro_prompt = st.sidebar.text_area("Pro Prompt", "You are arguing FOR the topic.")
    m_pro = st.sidebar.selectbox("Pro Model", list(models.keys()), index=0)
    
    st.sidebar.subheader("Con Side")
    con_prompt = st.sidebar.text_area("Con Prompt", "You are arguing AGAINST the topic.")
    m_con = st.sidebar.selectbox("Con Model", list(models.keys()), index=1)
    
    st.sidebar.subheader("Judge")
    judge_prompt = st.sidebar.text_area("Judge Prompt", "Review the debate and decide who won.")
    m_judge = st.sidebar.selectbox("Judge Model", list(models.keys()), index=0)
    
    rounds = st.sidebar.slider("Rounds:", 1, 3, 2)
    topic = st.text_input("Debate Topic:", "")
    
    if st.button("Start Debate"):
        if topic:
            with st.chat_message("user"):
                st.markdown(f"**Topic:** {topic}")
            transcript = f"Topic: {topic}\n\n"
            
            for i in range(1, rounds + 1):
                with st.chat_message("assistant", avatar="🟢"):
                    text_pro = st.empty()
                    pro_ans = ""
                    for chunk in stream_ai_response(pro_prompt, [{"role": "user", "content": transcript}], models[m_pro]):
                        pro_ans += chunk
                        text_pro.markdown(pro_ans)
                    transcript += f"PRO: {pro_ans}\n\n"

                with st.chat_message("assistant", avatar="🔴"):
                    text_con = st.empty()
                    con_ans = ""
                    for chunk in stream_ai_response(con_prompt, [{"role": "user", "content": transcript}], models[m_con]):
                        con_ans += chunk
                        text_con.markdown(con_ans)
                    transcript += f"CON: {con_ans}\n\n"

            st.markdown("---")
            with st.chat_message("assistant", avatar="⚖️"):
                text_judge = st.empty()
                verdict = ""
                for chunk in stream_ai_response(judge_prompt, [{"role": "user", "content": transcript}], models[m_judge]):
                    verdict += chunk
                    text_judge.markdown(verdict)