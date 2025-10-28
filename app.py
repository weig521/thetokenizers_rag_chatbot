## app.py
import streamlit as st
import os
from dotenv import load_dotenv
from utils.database import ChatDatabase
from utils.rag import generate_with_rag, estimate_tokens
from utils.security import sanitize_user_input, is_injection, AuthManager
from datetime import datetime

# Load environment variables
load_dotenv()

# Initialize clients / config
OLLAMA_URL      = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "phi3:mini")
CHROMA_PERSIST  = os.environ.get("CHROMA_PERSIST", "./data/processed")
CHROMA_COLLECTION = os.environ.get("CHROMA_COLLECTION", "usf_onboarding")
EMBED_MODEL     = os.environ.get("EMBED_MODEL", "google/embeddinggemma-300m")
SESSION_TOKEN_LIMIT = int(os.environ.get("SESSION_TOKEN_LIMIT", "1500"))  # total user+assistant tokens per session (small for testing)

db = ChatDatabase()

# Persist auth across reruns (st.session_state)
if "auth" not in st.session_state:
    st.session_state.auth = AuthManager(os.environ.get("AUTH_STORE_PATH", "./data/auth/users.json"))
auth = st.session_state.auth

# Page configuration, like the css/html
st.set_page_config(
    page_title="USF Onboarding Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .session-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "username" not in st.session_state:
    st.session_state.username = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_regen" not in st.session_state:
    st.session_state.pending_regen = False
# Token-budget tracking
if "token_total" not in st.session_state:
    st.session_state.token_total = 0
if "limit_reached" not in st.session_state:
    st.session_state.limit_reached = False

def _recompute_token_total(msgs: list[dict]) -> int:
    """Count only user+assistant tokens for the session budget."""
    return sum(
        estimate_tokens(m.get("content", ""))
        for m in msgs
        if m.get("role") in ("user", "assistant")
    )

# Login/Register Page
if not st.session_state.authenticated:
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.title("🤖 USF Onboarding Assistant")
        st.markdown("### Your AI-Powered Conversation Partner")

        tab1, tab2 = st.tabs(["Login", "Register"])

        with tab1:
            with st.form("login_form"):
                st.subheader("Welcome Back!")
                login_username = st.text_input("Username", key="login_username").strip()
                login_password = st.text_input("Password", type="password", key="login_password")
                submit = st.form_submit_button("Login", use_container_width=True, type="primary")

                if submit:
                    success, user_id = auth.authenticate_user(login_username, login_password)

                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_id = user_id
                        st.session_state.username = login_username
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

        with tab2:
            with st.form("register_form"):
                st.subheader("Create Account")
                reg_username = st.text_input("Username", key="reg_username").strip()
                reg_email    = st.text_input("Email (optional)", key="reg_email").strip()
                reg_password = st.text_input("Password", type="password", key="reg_password")
                reg_password2 = st.text_input("Confirm Password", type="password")
                submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")

                if submit:
                    if not reg_username or not reg_password:
                        st.error("Username and password are required")
                    elif reg_password != reg_password2:
                        st.error("Passwords don't match")
                    elif len(reg_password) < 6:
                        st.error("Password must be at least 6 characters")
                    else:
                        success, message = auth.create_user(reg_username, reg_password, reg_email)

                        if success:
                            st.success("Account created! Please login.")
                        else:
                            st.error(f"Registration failed: {message}")

# Main Application
else:
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {st.session_state.username}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚪 Logout", use_container_width=True):
                st.session_state.authenticated = False
                st.session_state.user_id = None
                st.session_state.username = None
                st.session_state.current_session_id = None
                st.session_state.messages = []
                st.session_state.token_total = 0
                st.session_state.limit_reached = False
                st.rerun()

        with col2:
            if st.button("⚙️ Settings", use_container_width=True):
                st.info("Settings coming soon!")

        st.divider()

        # New Session with exercise 2
        # Update messages

        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            session_name = f"Chat {datetime.now().strftime('%b %d, %H:%M')}"
            sid = db.create_session(st.session_state.user_id, session_name)
            st.session_state.current_session_id = sid
            st.session_state.messages = [
                {"role": "system", "content": "Assistant configured."}
            ]
            # Reset token budget for the new chat
            st.session_state.token_total = 0
            st.session_state.limit_reached = False
            st.rerun()

        st.divider()

        # Search
        search_query = st.text_input("🔍 Search sessions", key="search_input")

        # Filter sessions
        sessions = db.search_sessions(st.session_state.user_id, search_query)

        if sessions:
            st.markdown(f"### 📁 Sessions ({len(sessions)})")

            for session in sessions:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])

                    with col1:
                        is_current = session["session_id"] == st.session_state.current_session_id
                        button_type = "primary" if is_current else "secondary"

                        if st.button(
                            session["session_name"],
                            key=f"session_{session['session_id']}",
                            use_container_width=True,
                            type=button_type
                        ):
                            st.session_state.current_session_id = session["session_id"]
                            db_messages = db.get_session_messages(session["session_id"])
                            st.session_state.messages = [
                                {"role": "system", "content": "Assistant configured."}
                            ] + [
                                {"role": msg["role"], "content": msg["content"]}
                                for msg in db_messages
                            ]
                            # Recompute budget from loaded messages
                            st.session_state.token_total = _recompute_token_total(st.session_state.messages)
                            st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                            st.rerun()

                    with col2:
                        # Export button
                        export_json = db.export_session_json(st.session_state.user_id, session["session_id"])
                        st.download_button(
                            "📥",
                            data=export_json,
                            file_name=f"{session['session_name']}.json",
                            mime="application/json",
                            key=f"export_{session['session_id']}"
                        )

                    with col3:
                        # Delete button
                        if st.button("🗑️", key=f"delete_{session['session_id']}"):
                            db.delete_session(session["session_id"])
                            if st.session_state.current_session_id == session["session_id"]:
                                st.session_state.current_session_id = None
                                st.session_state.messages = []
                                st.session_state.token_total = 0
                                st.session_state.limit_reached = False
                            st.rerun()

                    # Session info
                    msg_count = len(db.get_session_messages(session["session_id"]))
                    st.caption(f"💬 {msg_count} messages • 🕒 {session['updated_at']}")
                    st.divider()
        else:
            st.info("No sessions found")

    # Main Chat Area
    if st.session_state.current_session_id:
        sessions = db.get_user_sessions(st.session_state.user_id)
        current_session = next(
            (s for s in sessions if s["session_id"] == st.session_state.current_session_id),
            None
        )

        if current_session:
            # Header
            col1, col2, col3 = st.columns([3, 1, 1])

            with col1:
                st.title(current_session['session_name'])

            with col2:
                # Stats
                msg_count = len(st.session_state.messages) - 1
                st.metric("Messages", msg_count)

            with col3:
                # Rename
                with st.popover("✏️ Options"):
                    new_name = st.text_input("Rename:", value=current_session['session_name'])
                    if st.button("Save", use_container_width=True):
                        db.rename_session(st.session_state.current_session_id, new_name)
                        st.success("Renamed!")
                        st.rerun()

                    st.divider()

                    # Export
                    export_json = db.export_session_json(st.session_state.user_id, st.session_state.current_session_id)
                    st.download_button(
                        "📥 Export Chat",
                        data=export_json,
                        file_name=f"{current_session['session_name']}.json",
                        mime="application/json",
                        use_container_width=True
                    )

        st.divider()

        # Display conversation
        for msg in st.session_state.messages[1:]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # excercise 4, Regeneration Button
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "assistant": # make sure msg generated and role is assistant
            if st.button("🔄 Regenerate Last Response"):
                st.session_state.messages.pop()  # Remove last assistant message from session state
                st.session_state.pending_regen = True
                st.rerun() # Rerunning triggers the API call logic
            
        # Auto-Regenerate path: if flagged and last message is user, re-ask with fresh retrieval
        if st.session_state.pending_regen and st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
                st.session_state.pending_regen = False
                last_user = st.session_state.messages[-1]["content"]
                with st.chat_message("assistant"):
                    placeholder = st.empty()
                    final_text = None
                    for kind, text in generate_with_rag(
                                                        last_user,
                                                        None,
                                                        CHROMA_PERSIST,
                                                        CHROMA_COLLECTION,
                                                        EMBED_MODEL,
                                                        OLLAMA_MODEL,
                                                        OLLAMA_URL,
                                                    ):
                        if kind == "delta":
                            placeholder.write(text)
                        else:
                            final_text = text
                            placeholder.write(final_text)
                out_toks = estimate_tokens(final_text or "")
                st.session_state.token_total += out_toks
                st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT

                st.session_state.messages.append({"role": "assistant", "content": final_text})
                db.add_message(st.session_state.current_session_id, "assistant", final_text)
                st.rerun()

        # User input (gated by token budget)
        if st.session_state.limit_reached:
            st.warning(
                f"Session token budget reached "
                f"({st.session_state.token_total}/{SESSION_TOKEN_LIMIT}). "
                "Please open a new session to continue."
            )
            user_input = None
        else:
            user_input = st.chat_input("Type your message here...")

        if user_input:
            clean = sanitize_user_input(user_input)

            # Store + echo the user message
            st.session_state.messages.append({"role": "user", "content": clean})
            db.add_message(st.session_state.current_session_id, "user", clean)
            with st.chat_message("user"):
                st.write(clean)

            # Block obvious injection before hitting the LLM
            if is_injection(clean):
                warn = "That looks like a prompt-injection attempt. For safety, I can’t run that. Try a normal question."
                with st.chat_message("assistant"):
                    st.write(warn)
                # Count tokens 
                in_toks  = estimate_tokens(clean)
                out_toks = estimate_tokens(warn)
                st.session_state.token_total += (in_toks + out_toks)
                st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT

                st.session_state.messages.append({"role": "assistant", "content": warn})
                db.add_message(st.session_state.current_session_id, "assistant", warn)
                st.rerun()

            # Stream a grounded answer via RAG
            with st.chat_message("assistant"):
                placeholder = st.empty()
                final_text = None
                for kind, text in generate_with_rag(
                                                    clean,
                                                    None,
                                                    CHROMA_PERSIST,
                                                    CHROMA_COLLECTION,
                                                    EMBED_MODEL,
                                                    OLLAMA_MODEL,
                                                    OLLAMA_URL,
                                                ):
                    if kind == "delta":
                        placeholder.write(text)
                    else:
                        final_text = text
                        placeholder.write(final_text)

            # Count tokens 
            in_toks  = estimate_tokens(clean)
            out_toks = estimate_tokens(final_text or "")
            st.session_state.token_total += (in_toks + out_toks)
            st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT

            st.session_state.messages.append({"role": "assistant", "content": final_text})
            db.add_message(st.session_state.current_session_id, "assistant", final_text)
            st.rerun()

    else:
        # Dashboard
        st.title("🤖 Welcome to USF Onboarding Assistant")
        st.markdown("### Your Personal AI Assistant")

        st.divider()

        # Stats
        sessions = db.get_user_sessions(st.session_state.user_id)

        col1, col2= st.columns(2)

        with col1:
            st.metric("📁 Total Sessions", len(sessions))

        with col2:
            total_messages = sum(len(db.get_session_messages(s["session_id"])) for s in sessions)
            st.metric("💬 Total Messages", total_messages)

        st.divider()

        # Recent sessions
        if sessions:
            st.subheader("📌 Recent Sessions")

            for session in sessions[:5]:
                with st.expander(f"💬 {session['session_name']}", expanded=False):
                    messages = db.get_session_messages(session["session_id"])

                    st.caption(f"Created: {session['created_at']}")
                    st.caption(f"Updated: {session['updated_at']}")
                    st.caption(f"Messages: {len(messages)}")

                    if st.button("Open", key=f"open_{session['session_id']}"):
                        st.session_state.current_session_id = session["session_id"]
                        st.session_state.messages = [
                            {"role": "system", "content": "Assistant configured."}
                        ] + [
                            {"role": msg["role"], "content": msg["content"]}
                            for msg in messages
                        ]
                        st.session_state.token_total = _recompute_token_total(st.session_state.messages)
                        st.session_state.limit_reached = st.session_state.token_total >= SESSION_TOKEN_LIMIT
                        st.rerun()

        else:
            st.info("👈 Create your first session to start chatting!")
