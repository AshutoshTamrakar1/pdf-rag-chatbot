import reflex as rx

config = rx.Config(
    app_name="app",
    backend_port=8001,  # Reflex backend on port 8001 to avoid conflict with FastAPI on 8000
    frontend_port=3000,
)
