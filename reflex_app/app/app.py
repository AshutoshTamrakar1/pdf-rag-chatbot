import reflex as rx
import httpx
from typing import List, Dict
import asyncio

BACKEND_URL = "http://localhost:8000"

# Shared authentication data (simple approach for state sharing)
class SharedAuth(rx.Base):
    token: str = ""
    session_id: str = ""
    user_id: str = ""

class AuthState(rx.State):
    username: str = ""
    password: str = ""
    email: str = ""
    confirm_password: str = ""
    token: str = ""
    session_id: str = ""
    user_id: str = ""
    error: str = ""
    success: str = ""
    
    async def login(self):
        """Login user via backend API"""
        print(f"[LOGIN] Attempting login for email: {self.email}")
        self.error = ""
        self.success = ""
        
        if not self.email or not self.password:
            self.error = "Email and password are required"
            print(f"[LOGIN ERROR] {self.error}")
            return
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print(f"[LOGIN] Sending request to {BACKEND_URL}/auth/login")
                response = await client.post(
                    f"{BACKEND_URL}/auth/login",
                    json={"email": self.email, "password": self.password}
                )
                
                print(f"[LOGIN] Response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    self.token = data["access_token"]
                    self.session_id = data["session_id"]
                    self.user_id = data["user_id"]
                    self.success = "Login successful!"
                    print(f"[LOGIN SUCCESS] User {self.user_id} logged in, session: {self.session_id}")
                    
                    # Redirect with session info in URL (will be picked up by MainState)
                    yield rx.redirect(f"/chat?session_id={self.session_id}&user_id={self.user_id}&token={self.token}")
                else:
                    error_data = response.json()
                    self.error = error_data.get("detail", "Login failed")
                    print(f"[LOGIN ERROR] {self.error}")
        except Exception as e:
            self.error = f"Connection error: {str(e)}"
            print(f"[LOGIN EXCEPTION] {e}")
    
    async def register(self):
        """Register new user via backend API"""
        print(f"[REGISTER] Attempting registration for email: {self.email}")
        self.error = ""
        self.success = ""
        
        if not self.email or not self.username or not self.password or not self.confirm_password:
            self.error = "All fields are required"
            print(f"[REGISTER ERROR] {self.error}")
            return
        
        if self.password != self.confirm_password:
            self.error = "Passwords do not match"
            print(f"[REGISTER ERROR] {self.error}")
            return
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print(f"[REGISTER] Sending request to {BACKEND_URL}/auth/register")
                response = await client.post(
                    f"{BACKEND_URL}/auth/register",
                    json={
                        "email": self.email,
                        "username": self.username,
                        "password": self.password,
                        "confirm_password": self.confirm_password
                    }
                )
                
                print(f"[REGISTER] Response status: {response.status_code}")
                if response.status_code == 201:
                    data = response.json()
                    self.success = f"Registration successful! Welcome {data['username']}"
                    print(f"[REGISTER SUCCESS] User {data['user_id']} registered")
                    # Clear form
                    self.email = ""
                    self.username = ""
                    self.password = ""
                    self.confirm_password = ""
                    # Redirect to login after 2 seconds
                    yield rx.call_script("setTimeout(() => window.location.href = '/', 2000)")
                else:
                    error_data = response.json()
                    self.error = error_data.get("detail", "Registration failed")
                    print(f"[REGISTER ERROR] {self.error}")
        except Exception as e:
            self.error = f"Connection error: {str(e)}"
            print(f"[REGISTER EXCEPTION] {e}")


class MainState(rx.State):
    # Authentication - will be set when redirected from login
    token: str = ""
    session_id: str = ""
    user_id: str = ""
    
    # Chat sessions
    chat_sessions: list[dict] = []
    current_chat_id: str = ""
    current_chat_title: str = "New Chat"
    
    # Messages
    messages: list[dict] = []
    user_input: str = ""
    
    # PDFs
    uploaded_pdfs: list[dict] = []
    active_source_ids: list[str] = []
    
    # UI state
    error: str = ""
    is_loading: bool = False
    show_upload: bool = False
    
    # Model selection
    available_models: list[str] = []
    selected_model: str = "llama3"
    
    def set_auth_from_login(self, token: str, session_id: str, user_id: str):
        """Called to set auth data from login page"""
        self.token = token
        self.session_id = session_id
        self.user_id = user_id
        print(f"[MAIN] Auth set - session_id: {self.session_id}")
    
    async def on_load(self):
        """Load user data when chat page loads"""
        print("[MAIN] Loading chat page...")
        
        # Get auth from URL query params (passed from login)
        if self.router.page.params:
            new_session = self.router.page.params.get("session_id", "")
            if new_session and new_session != self.session_id:
                # New login detected - clear old state
                print(f"[MAIN] New session detected, clearing old state")
                self.session_id = new_session
                self.user_id = self.router.page.params.get("user_id", "")
                self.token = self.router.page.params.get("token", "")
                self.current_chat_id = ""  # Clear old chat
                self.messages = []
                self.chat_sessions = []
                self.uploaded_pdfs = []
                self.active_source_ids = []
                print(f"[MAIN] Auth loaded from URL params - session_id: {self.session_id[:20] if self.session_id else 'None'}...")
        
        # Load available models
        await self.load_available_models()
        
        print(f"[MAIN] Current auth - session_id: {self.session_id[:20] if self.session_id else 'None'}...")
        
        # If no session, user needs to login
        if not self.session_id:
            print("[MAIN] No session, redirecting to login")
            yield rx.redirect("/")
            return
        
        # Create initial chat session if none exists
        if not self.current_chat_id:
            print("[MAIN] Creating initial chat session")
            # Use async for to handle generator
            async for _ in self.create_new_chat():
                pass
    
    async def create_new_chat(self):
        """Create a new chat session"""
        print("[CREATE_CHAT] Creating new chat session")
        self.error = ""
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                print(f"[CREATE_CHAT] Sending request to {BACKEND_URL}/pdf/session")
                response = await client.post(
                    f"{BACKEND_URL}/pdf/session",
                    data={"session_id": self.session_id}
                )
                
                print(f"[CREATE_CHAT] Response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    self.current_chat_id = data["chat_session_id"]
                    self.current_chat_title = data.get("title", "New Chat")
                    self.messages = []
                    self.uploaded_pdfs = []
                    self.active_source_ids = []
                    
                    # Add to sessions list
                    self.chat_sessions.append({
                        "id": self.current_chat_id,
                        "title": self.current_chat_title
                    })
                    print(f"[CREATE_CHAT SUCCESS] Created chat session: {self.current_chat_id}")
                elif response.status_code == 403:
                    # Session expired or invalid
                    self.error = "Session expired. Please log in again."
                    print("[CREATE_CHAT ERROR] Session expired")
                    yield rx.redirect("/")
                else:
                    self.error = "Failed to create chat session"
                    print(f"[CREATE_CHAT ERROR] {self.error}")
        except Exception as e:
            self.error = f"Error creating chat: {str(e)}"
            print(f"[CREATE_CHAT EXCEPTION] {e}")
    
    async def upload_pdf(self, files: list):
        """Upload PDF file"""
        print(f"[UPLOAD] Uploading {len(files)} file(s)")
        self.error = ""
        self.is_loading = True
        
        if not files:
            self.error = "No file selected"
            self.is_loading = False
            return
        
        try:
            file = files[0]
            filename = file.filename if hasattr(file, 'filename') else str(file)
            print(f"[UPLOAD] Processing file: {filename}")
            
            # Read file content
            file_data = await file.read() if hasattr(file, 'read') else file
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                files_data = {
                    "file": (filename, file_data, "application/pdf")
                }
                form_data = {
                    "session_id": self.session_id,
                    "chat_session_id": self.current_chat_id
                }
                
                print(f"[UPLOAD] Sending to {BACKEND_URL}/pdf/upload")
                response = await client.post(
                    f"{BACKEND_URL}/pdf/upload",
                    data=form_data,
                    files=files_data
                )
                
                print(f"[UPLOAD] Response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    new_source = data["new_source"]
                    self.uploaded_pdfs.append(new_source)
                    self.active_source_ids.append(new_source["source_id"])
                    
                    # Update title if changed
                    if data.get("new_title"):
                        self.current_chat_title = data["new_title"]
                    
                    self.show_upload = False
                    print(f"[UPLOAD SUCCESS] PDF uploaded: {new_source['filename']}")
                else:
                    self.error = "Upload failed"
                    print(f"[UPLOAD ERROR] {self.error}")
        except Exception as e:
            self.error = f"Upload error: {str(e)}"
            print(f"[UPLOAD EXCEPTION] {e}")
        finally:
            self.is_loading = False
    
    async def handle_upload(self, files: list[rx.UploadFile]):
        """Handle file upload from rx.upload component"""
        print(f"[HANDLE_UPLOAD] Files received: {files}")
        await self.upload_pdf(files)
    
    async def send_message(self):
        """Send chat message"""
        print(f"[SEND_MSG] Sending message: {self.user_input[:50]}...")
        
        if not self.user_input.strip():
            print("[SEND_MSG] Empty message, skipping")
            return
        
        self.error = ""
        user_msg = self.user_input
        self.user_input = ""
        
        # Add user message to UI immediately
        self.messages.append({"role": "user", "content": user_msg})
        self.is_loading = True
        
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                # Convert active_source_ids to plain list to avoid serialization issues
                source_ids = list(self.active_source_ids) if self.active_source_ids else []
                
                payload = {
                    "session_id": self.session_id,
                    "chat_session_id": self.current_chat_id,
                    "user_input": user_msg,
                    "active_source_ids": source_ids,
                    "model": self.selected_model
                }
                
                print(f"[SEND_MSG] Payload: {payload}")
                print(f"[SEND_MSG] Sending to {BACKEND_URL}/chat/send")
                
                response = await client.post(
                    f"{BACKEND_URL}/chat/send",
                    json=payload
                )
                
                print(f"[SEND_MSG] Response status: {response.status_code}")
                if response.status_code == 200:
                    data = response.json()
                    assistant_msg = data["answer"]
                    self.messages.append({"role": "assistant", "content": assistant_msg})
                    print(f"[SEND_MSG SUCCESS] Got response: {assistant_msg[:50]}...")
                    
                    # Refresh chat title if this was the first message
                    if self.current_chat_title == "New Chat" and len(self.messages) == 2:
                        await self.refresh_chat_title()
                else:
                    error_data = response.json()
                    self.error = error_data.get("detail", "Failed to send message")
                    print(f"[SEND_MSG ERROR] {self.error}")
        except Exception as e:
            self.error = f"Error: {str(e)}"
            print(f"[SEND_MSG EXCEPTION] {e}")
        finally:
            self.is_loading = False
    
    def toggle_upload(self):
        """Toggle upload modal"""
        self.show_upload = not self.show_upload
        print(f"[UI] Upload modal: {self.show_upload}")
    
    async def load_available_models(self):
        """Load available chat models from backend"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{BACKEND_URL}/chat/models")
                if response.status_code == 200:
                    data = response.json()
                    self.available_models = data.get("models", ["llama3", "gemma", "phi3"])
                    self.selected_model = data.get("default", "llama3")
                    print(f"[MODELS] Loaded: {self.available_models}")
        except Exception as e:
            print(f"[MODELS ERROR] {e}")
            self.available_models = ["llama3", "gemma", "phi3"]
    
    def set_model(self, model: str):
        """Set selected chat model"""
        self.selected_model = model
        print(f"[MODEL] Selected: {model}")
    
    def logout(self):
        """Logout user and clear session"""
        print(f"[LOGOUT] Logging out user")
        self.session_id = ""
        self.user_id = ""
        self.token = ""
        self.current_chat_id = ""
        self.messages = []
        self.chat_sessions = []
        self.uploaded_pdfs = []
        self.active_source_ids = []
        return rx.redirect("/")
    
    async def refresh_chat_title(self):
        """Fetch updated chat title from backend"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{BACKEND_URL}/pdf/session/{self.current_chat_id}",
                    params={"session_id": self.session_id}
                )
                if response.status_code == 200:
                    data = response.json()
                    new_title = data.get("title", "New Chat")
                    if new_title != "New Chat":
                        self.current_chat_title = new_title
                        # Update in sessions list
                        for session in self.chat_sessions:
                            if session["id"] == self.current_chat_id:
                                session["title"] = new_title
                        print(f"[TITLE] Updated to: {new_title}")
        except Exception as e:
            print(f"[TITLE ERROR] {e}")


def login_page():
    return rx.container(
        rx.vstack(
            rx.heading("Login", size="9"),
            rx.input(
                placeholder="Email",
                value=AuthState.email,
                on_change=AuthState.set_email,
            ),
            rx.input(
                placeholder="Password",
                type="password",
                value=AuthState.password,
                on_change=AuthState.set_password,
            ),
            rx.button("Login", on_click=AuthState.login),
            rx.text(AuthState.success, color="green"),
            rx.text(AuthState.error, color="red"),
            rx.text("Don't have an account?"),
            rx.button("Register", on_click=lambda: rx.redirect("/register")),
            spacing="4",
            align="center",
        ),
        padding="4em",
    )

def register_page():
    return rx.container(
        rx.vstack(
            rx.heading("Register", size="9"),
            rx.input(
                placeholder="Email",
                value=AuthState.email,
                on_change=AuthState.set_email,
            ),
            rx.input(
                placeholder="Username",
                value=AuthState.username,
                on_change=AuthState.set_username,
            ),
            rx.input(
                placeholder="Password",
                type="password",
                value=AuthState.password,
                on_change=AuthState.set_password,
            ),
            rx.input(
                placeholder="Confirm Password",
                type="password",
                value=AuthState.confirm_password,
                on_change=AuthState.set_confirm_password,
            ),
            rx.button("Register", on_click=AuthState.register),
            rx.text(AuthState.success, color="green"),
            rx.text(AuthState.error, color="red"),
            rx.text("Already have an account?"),
            rx.button("Login", on_click=lambda: rx.redirect("/")),
            spacing="4",
            align="center",
        ),
        padding="4em",
    )

def chat_page():
    """Main chat interface with sidebar"""
    return rx.box(
        # Sidebar
        rx.box(
            rx.vstack(
                rx.heading("Chats", size="6"),
                rx.button(
                    "+ New Chat",
                    on_click=MainState.create_new_chat,
                    width="100%"
                ),
                rx.divider(),
                rx.foreach(
                    MainState.chat_sessions,
                    lambda session: rx.box(
                        rx.text(session["title"]),
                        padding="0.5em",
                        _hover={"bg": "gray.100", "cursor": "pointer"},
                    )
                ),
                rx.spacer(),
                rx.divider(),
                rx.text("Uploaded PDFs:", font_weight="bold", size="2"),
                rx.foreach(
                    MainState.uploaded_pdfs,
                    lambda pdf: rx.box(
                        rx.text(pdf["filename"], size="1"),
                        padding="0.25em",
                        bg="blue.50",
                        border_radius="0.25em",
                        margin="0.25em 0",
                    )
                ),
                rx.divider(),
                rx.button(
                    "Logout",
                    on_click=MainState.logout,
                    width="100%",
                    color_scheme="red",
                    variant="outline"
                ),
                spacing="2",
                width="100%",
                height="100%",
            ),
            width="250px",
            height="100vh",
            bg="gray.50",
            padding="1em",
            position="fixed",
            left="0",
            top="0",
            display="flex",
            flex_direction="column",
        ),
        # Main chat area
        rx.box(
            rx.vstack(
                # Header
                rx.hstack(
                    rx.heading(MainState.current_chat_title, size="7"),
                    rx.spacer(),
                    rx.button(
                        "📎 Upload PDF",
                        on_click=MainState.toggle_upload,
                    ),
                    width="100%",
                    padding="1em",
                    border_bottom="1px solid #e2e8f0",
                ),
                # Messages
                rx.box(
                    rx.foreach(
                        MainState.messages,
                        lambda msg: rx.box(
                            rx.text(
                                msg["content"],
                                bg=rx.cond(
                                    msg["role"] == "user",
                                    "blue.100",
                                    "gray.100"
                                ),
                                padding="1em",
                                border_radius="0.5em",
                                max_width="70%",
                                margin_left=rx.cond(
                                    msg["role"] == "user",
                                    "auto",
                                    "0"
                                ),
                            ),
                            width="100%",
                            padding="0.5em",
                        )
                    ),
                    flex="1",
                    overflow_y="auto",
                    padding="1em",
                    width="100%",
                ),
                # Error message
                rx.cond(
                    MainState.error != "",
                    rx.text(MainState.error, color="red", padding="0.5em"),
                ),
                # Model selector
                rx.hstack(
                    rx.text("Model:", font_weight="500"),
                    rx.select(
                        MainState.available_models,
                        value=MainState.selected_model,
                        on_change=MainState.set_model,
                    ),
                    padding="0.5em 1em",
                    spacing="2",
                    border_top="1px solid #e2e8f0",
                    width="100%",
                ),
                # Input area
                rx.hstack(
                    rx.input(
                        placeholder="Type your message...",
                        value=MainState.user_input,
                        on_change=MainState.set_user_input,
                        flex="1",
                    ),
                    rx.button(
                        "Send",
                        on_click=MainState.send_message,
                        is_loading=MainState.is_loading,
                    ),
                    width="100%",
                    padding="1em",
                    border_top="1px solid #e2e8f0",
                ),
                height="100vh",
                width="100%",
                spacing="0",
            ),
            margin_left="250px",
        ),
        # Upload modal
        rx.cond(
            MainState.show_upload,
            rx.box(
                rx.box(
                    rx.vstack(
                        rx.heading("Upload PDF", size="6"),
                        rx.upload(
                            rx.button("Select PDF File"),
                            id="pdf_upload",
                            accept={"application/pdf": [".pdf"]},
                        ),
                        rx.button(
                            "Upload",
                            on_click=lambda: MainState.handle_upload(rx.upload_files(upload_id="pdf_upload")),
                        ),
                        rx.button(
                            "Close",
                            on_click=MainState.toggle_upload,
                        ),
                        spacing="3",
                    ),
                    bg="white",
                    padding="2em",
                    border_radius="0.5em",
                    box_shadow="lg",
                ),
                position="fixed",
                top="50%",
                left="50%",
                transform="translate(-50%, -50%)",
                bg="rgba(0,0,0,0.5)",
                width="100vw",
                height="100vh",
                z_index="1000",
            )
        ),
        on_mount=MainState.on_load,
    )

app = rx.App()
app.add_page(login_page, route="/")
app.add_page(register_page, route="/register")
app.add_page(chat_page, route="/chat")
