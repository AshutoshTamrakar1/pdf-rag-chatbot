"""
PDF RAG Chatbot - Streamlit Frontend
A single-page app for uploading PDFs, chatting, and generating mindmaps & podcasts.
"""

import streamlit as st
import requests
import json
import os
from datetime import datetime

# Configuration
API_BASE = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = 30

# Page config
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .chat-message { padding: 12px; margin: 8px 0; border-radius: 8px; }
    .chat-me { background-color: #0f3460; color: white; margin-left: 40px; }
    .chat-bot { background-color: #16213e; color: #e0e0e0; margin-right: 40px; }
    .chat-system { background-color: #1a1a2e; color: #888; margin: 8px 0; }
    .status-online { color: #00ff00; }
    .status-offline { color: #ff0000; }
    .error-box { background-color: #8b0000; padding: 10px; border-radius: 5px; color: white; }
    .success-box { background-color: #006400; padding: 10px; border-radius: 5px; color: white; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "user_id" not in st.session_state:
    st.session_state.user_id = None
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "chat_session_id" not in st.session_state:
    st.session_state.chat_session_id = None
if "uploaded_sources" not in st.session_state:
    st.session_state.uploaded_sources = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "generated_podcast" not in st.session_state:
    st.session_state.generated_podcast = None


def api_request(method, endpoint, data=None, files=None, timeout=None):
    """Make API request with error handling"""
    if timeout is None:
        timeout = API_TIMEOUT
    
    headers = {}
    if st.session_state.token:
        headers["Authorization"] = f"Bearer {st.session_state.token}"
    
    url = f"{API_BASE}{endpoint}"
    
    try:
        if method == "GET":
            resp = requests.get(url, headers=headers, timeout=timeout)
        elif method == "POST":
            if files:
                resp = requests.post(url, headers=headers, files=files, data=data, timeout=timeout)
            else:
                headers["Content-Type"] = "application/json"
                resp = requests.post(url, headers=headers, json=data, timeout=timeout)
        else:
            return None, "Unknown method"
        
        if resp.status_code in [200, 201]:
            try:
                result = resp.json()
                # Log response for debugging long operations
                if endpoint.startswith("/mindmap"):
                    st.write(f"✅ Response received from {endpoint}")
                return result, None
            except Exception as e:
                return resp.text, None
        else:
            # For 422 errors, include full detail
            try:
                error_detail = resp.json()
                return None, f"Error {resp.status_code}: {error_detail}"
            except:
                return None, f"Error {resp.status_code}: {resp.text[:200]}"
    except requests.exceptions.Timeout as te:
        return None, f"Request timeout after {timeout}s - Operation may still be processing on server"
    except requests.exceptions.ConnectionError as ce:
        return None, f"Connection error: {str(ce)}"
    except Exception as e:
        return None, str(e)


def login_page():
    """Login/Register UI"""
    st.title("🤖 PDF RAG Chatbot")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login")
        email = st.text_input("Email (login)", key="login_email")
        password = st.text_input("Password (login)", type="password", key="login_pwd")
        
        if st.button("Login", key="login_btn"):
            if email and password:
                result, err = api_request("POST", "/auth/login", {"email": email, "password": password})
                if result and result.get("access_token"):
                    st.session_state.token = result["access_token"]
                    st.session_state.session_id = result.get("session_id")  # Store backend session_id
                    st.session_state.user_id = result.get("user_id")
                    st.session_state.chat_messages = []
                    
                    # Auto-create a chat session on login
                    headers = {"Authorization": f"Bearer {result['access_token']}"}
                    url = f"{API_BASE}/pdf/session"
                    try:
                        resp = requests.post(url, headers=headers, data={"session_id": result["session_id"]}, timeout=API_TIMEOUT)
                        if resp.status_code in [200, 201]:
                            session_result = resp.json()
                            st.session_state.chat_session_id = session_result.get("chat_session_id")
                    except:
                        pass  # Silently fail if session creation fails
                    
                    st.success("✅ Logged in successfully!")
                    st.rerun()
                else:
                    st.error(f"❌ Login failed: {err}")
            else:
                st.warning("⚠️ Enter email and password")
    
    with col2:
        st.subheader("Register")
        reg_username = st.text_input("Username", key="reg_user")
        reg_email = st.text_input("Email (register)", key="reg_email")
        reg_password = st.text_input("Password (register)", type="password", key="reg_pwd")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
        
        if st.button("Register", key="reg_btn"):
            if reg_username and reg_email and reg_password and reg_confirm:
                result, err = api_request("POST", "/auth/register", {
                    "username": reg_username,
                    "email": reg_email,
                    "password": reg_password,
                    "confirm_password": reg_confirm
                })
                if result and result.get("status") == "success":
                    st.success("✅ Registration successful! Please login.")
                else:
                    st.error(f"❌ Registration failed: {err}")
            else:
                st.warning("⚠️ Fill all fields")


def chat_page():
    """Main chat interface"""
    st.title("🤖 PDF RAG Chatbot")
    
    # Sidebar: session & uploads
    with st.sidebar:
        st.subheader("📁 Session Management")
        
        if st.button("➕ New Chat Session", key="new_session"):
            # Create a new chat session on backend
            headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
            url = f"{API_BASE}/pdf/session"
            
            try:
                resp = requests.post(url, headers=headers, data={"session_id": st.session_state.session_id}, timeout=API_TIMEOUT)
                if resp.status_code in [200, 201]:
                    result = resp.json()
                    st.session_state.chat_session_id = result.get("chat_session_id")
                    st.session_state.chat_messages = []
                    st.session_state.uploaded_sources = []
                    st.success(f"✅ New session created!")
                else:
                    st.error(f"❌ Failed to create session: Error {resp.status_code}")
            except Exception as e:
                st.error(f"❌ Error creating session: {str(e)}")
        
        # Display current session info
        if st.session_state.chat_session_id:
            st.info(f"📋 Session ID: {st.session_state.chat_session_id[:8]}...")
        
        st.markdown("---")
        st.subheader("📄 Uploaded PDFs")
        
        uploaded_file = st.file_uploader("Upload PDF", type="pdf", key="pdf_upload")
        if uploaded_file:
            with st.spinner("Uploading..."):
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                }
                data = {
                    "session_id": st.session_state.session_id,
                    "chat_session_id": st.session_state.chat_session_id
                }
                
                headers = {"Authorization": f"Bearer {st.session_state.token}"} if st.session_state.token else {}
                url = f"{API_BASE}/pdf/upload"
                
                try:
                    resp = requests.post(url, headers=headers, files=files, data=data, timeout=API_TIMEOUT)
                    if resp.status_code in [200, 201]:
                        result = resp.json()
                        new_source = result.get("new_source", {})
                        st.session_state.uploaded_sources.append({
                            "id": new_source.get("source_id"),
                            "name": new_source.get("filename", uploaded_file.name),
                            "size": 0
                        })
                        st.success(f"✅ Uploaded: {uploaded_file.name}")
                    else:
                        st.error(f"❌ Upload failed: Error {resp.status_code}: {resp.text[:200]}")
                except Exception as e:
                    st.error(f"❌ Upload failed: {str(e)}")
        
        if st.session_state.uploaded_sources:
            st.write("**Uploaded files:**")
            for src in st.session_state.uploaded_sources:
                st.write(f"📌 {src['name']} ({src['size'] / 1024:.1f} KB)")
        
        st.markdown("---")
        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.token = None
            st.session_state.user_id = None
            st.session_state.chat_messages = []
            st.success("Logged out")
            st.rerun()
    
    # Main content: Chat
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("💬 Chat")
    with col2:
        status = "🟢 Online" if st.session_state.token else "🔴 Offline"
        st.markdown(f"<p style='text-align:right'>{status}</p>", unsafe_allow_html=True)
    
    # Chat message history
    st.markdown("### Chat Messages")
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='chat-message chat-me'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
            elif msg["role"] == "bot":
                st.markdown(f"<div class='chat-message chat-bot'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='chat-message chat-system'><i>{msg['content']}</i></div>", unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    user_input = st.text_area("Your message:", height=80, key="user_msg")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        if st.button("📤 Send Message", key="send_msg"):
            if user_input.strip():
                # Validate session before sending
                if not st.session_state.session_id:
                    st.error("❌ Error: No session ID. Please login again.")
                elif not st.session_state.chat_session_id:
                    st.error("❌ Error: No chat session. Create a new chat first.")
                else:
                    # Add user message
                    st.session_state.chat_messages.append({"role": "user", "content": user_input})
                    
                    # Call backend
                    with st.spinner("Waiting for response..."):
                        payload = {
                            "session_id": st.session_state.session_id,
                            "chat_session_id": st.session_state.chat_session_id,
                            "user_input": user_input,
                            "active_source_ids": [s["id"] for s in st.session_state.uploaded_sources]
                        }
                        result, err = api_request("POST", "/chat/send", payload)
                        
                        if result:
                            response = result.get("answer", "No response")
                            st.session_state.chat_messages.append({"role": "bot", "content": response})
                            st.success("✅ Response received!")
                            st.rerun()
                        else:
                            st.error(f"❌ Error: {err}")
                
                st.rerun()
    
    with col2:
        if st.button("🧠 Generate Mindmap", key="gen_mindmap"):
            if st.session_state.uploaded_sources:
                with st.spinner("Generating mindmap..."):
                    # Generate mindmap for the first source
                    source = st.session_state.uploaded_sources[0]
                    result, err = api_request("POST", "/mindmap/", {
                        "session_id": st.session_state.session_id,
                        "chat_session_id": st.session_state.chat_session_id,
                        "thread_id": st.session_state.chat_session_id,  # Backward compatibility
                        "source_id": source["id"]
                    })
                    
                    if result:
                        mindmap = result.get("markdown", "")
                        st.session_state.chat_messages.append({
                            "role": "system",
                            "content": f"📊 Mindmap generated:\n```\n{mindmap}\n```"
                        })
                        st.success("✅ Mindmap generated!")
                        st.rerun()
                    else:
                        st.error(f"❌ Failed: {err}")
            else:
                st.warning("⚠️ Upload a PDF first")
    
    with col3:
        if st.button("🎙️ Generate Podcast", key="gen_podcast"):
            if st.session_state.uploaded_sources:
                with st.spinner("Generating podcast (this may take 1-2 minutes)..."):
                    # Generate podcast for the first source
                    source = st.session_state.uploaded_sources[0]
                    result, err = api_request("POST", "/mindmap/podcast", {
                        "session_id": st.session_state.session_id,
                        "chat_session_id": st.session_state.chat_session_id,
                        "thread_id": st.session_state.chat_session_id,  # Backward compatibility
                        "source_id": source["id"]
                    }, timeout=300)  # 5 minute timeout for podcast generation
                    
                    if result:
                        podcast_data = result.get("data", {})
                        audio_base64 = podcast_data.get("audio_base64", "") if isinstance(podcast_data, dict) else ""
                        script = podcast_data.get("script", "") if isinstance(podcast_data, dict) else ""
                        
                        if audio_base64:
                            # Store podcast in session state for persistent display
                            st.session_state.generated_podcast = {
                                "audio_base64": audio_base64,
                                "script": script,
                                "generated_at": podcast_data.get("generated_at", "")
                            }
                            
                            # Add message to chat
                            st.session_state.chat_messages.append({
                                "role": "system",
                                "content": f"🎙️ Podcast generated! Script length: {len(script)} characters"
                            })
                            st.success("✅ Podcast generated successfully!")
                        else:
                            st.error("❌ Podcast audio was not generated. Please try again.")
                    else:
                        st.error(f"❌ Podcast generation failed: {err}")
            else:
                st.warning("⚠️ Upload a PDF first")
    
    # Display generated podcast if available
    if st.session_state.generated_podcast:
        st.markdown("---")
        st.markdown("### 🎧 Podcast Audio")
        
        # Create audio player with base64 data
        audio_base64 = st.session_state.generated_podcast.get("audio_base64", "")
        script = st.session_state.generated_podcast.get("script", "")
        
        if audio_base64:
            audio_html = f"""
            <audio controls style="width: 100%;">
                <source src="data:audio/wav;base64,{audio_base64}" type="audio/wav">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
        
        # Display script preview
        if script:
            with st.expander("📜 View Podcast Script"):
                st.text(script)


def main():
    """Main app logic"""
    if st.session_state.token:
        chat_page()
    else:
        login_page()


if __name__ == "__main__":
    main()
