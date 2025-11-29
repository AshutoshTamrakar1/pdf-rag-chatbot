import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator
import jwt as pyjwt
import bcrypt
from logging_config import get_logger, log_exceptions
from db_manager import (
    create_user,
    get_user_by_id,
    update_user_last_login,
    create_session as create_session_in_db,
    get_session as get_session_from_db,
    COLLECTION_USERS
)

logger = get_logger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480"))  # 8 hours
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))  # 7 days

# Active sessions in memory (for quick lookup)
# In production, use Redis or persistent store
active_sessions: Dict[str, Dict[str, Any]] = {}

# ============================================================================
# PYDANTIC MODELS
#
# NOTE: This project uses Pydantic v2 style validators. The old `validator`
# decorator from Pydantic v1 is deprecated — we use `field_validator` for
# single-field validation and `model_validator` for cross-field checks (e.g.
# password/confirm_password equality).
# ============================================================================

class UserRegisterRequest(BaseModel):
    """User registration request model"""
    email: EmailStr
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=128)
    confirm_password: str = Field(..., min_length=8, max_length=128)

    # Pydantic v2: use field_validator for single-field validation
    @field_validator("username")
    def validate_username(cls, v):
        """Validate username format"""
        # allow alphanumeric characters or underscore or hyphen
        if not v.isalnum() and "_" not in v and "-" not in v:
            raise ValueError("Username can only contain alphanumeric characters, underscores, and hyphens")
        return v

    # Confirm password must match password — this is a cross-field validation
    @model_validator(mode="after")
    def validate_passwords_match(cls, model):
        """Ensure passwords match (model-level validator)"""
        if getattr(model, "password", None) is not None and getattr(model, "confirm_password", None) != model.password:
            raise ValueError("Passwords do not match")
        return model


class UserLoginRequest(BaseModel):
    """User login request model"""
    email: EmailStr
    password: str


class UserLoginResponse(BaseModel):
    """User login response model"""
    status: str
    message: str
    user_id: str
    session_id: str
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "bearer"
    expires_in: int  # seconds


class UserRegisterResponse(BaseModel):
    """User registration response model"""
    status: str
    message: str
    user_id: str
    email: str
    username: str


class TokenRefreshRequest(BaseModel):
    """Token refresh request model"""
    refresh_token: str


class TokenRefreshResponse(BaseModel):
    """Token refresh response model"""
    status: str
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class UserProfileResponse(BaseModel):
    """User profile response model"""
    user_id: str
    email: str
    username: str
    created_at: str
    last_login: Optional[str]
    active: bool


# ============================================================================
# PASSWORD HASHING
# ============================================================================

def hash_password(password: str) -> str:
    """Hash password using bcrypt"""
    salt = bcrypt.gensalt(rounds=12)
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify password against hash"""
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8")
    )


# ============================================================================
# JWT TOKEN MANAGEMENT
# ============================================================================

def create_access_token(user_id: str, expires_delta: Optional[timedelta] = None) -> tuple[str, int]:
    """Create JWT access token"""
    if expires_delta is None:
        expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    expire = datetime.utcnow() + expires_delta
    payload = {
        "sub": user_id,
        "type": "access",
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    token = pyjwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    expires_in = int(expires_delta.total_seconds())
    
    logger.info(f"Access token created for user: {user_id}")
    return token, expires_in


def create_refresh_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """Create JWT refresh token"""
    if expires_delta is None:
        expires_delta = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    expire = datetime.utcnow() + expires_delta
    payload = {
        "sub": user_id,
        "type": "refresh",
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    token = pyjwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    logger.info(f"Refresh token created for user: {user_id}")
    return token


def verify_token(token: str, token_type: str = "access") -> Optional[str]:
    """Verify JWT token and return user_id"""
    try:
        payload = pyjwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        token_type_in_payload = payload.get("type")
        
        if user_id is None or token_type_in_payload != token_type:
            logger.warning(f"Invalid token type or missing user_id")
            return None
        
        return user_id
    except pyjwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except pyjwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None


# ============================================================================
# SESSION MANAGEMENT
# ============================================================================

def create_session(user_id: str) -> str:
    """Create a new session for user (stores both in-memory and database)"""
    session_id = str(uuid.uuid4())
    
    session_data = {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat(),
        "last_activity": datetime.utcnow().isoformat(),
        "active": True
    }
    
    # Store in memory for fast access
    active_sessions[session_id] = session_data
    
    # Store in database for persistence
    try:
        create_session_in_db(session_data)
    except Exception as e:
        logger.error(f"Failed to store session in database: {e}")
    
    logger.info(f"Session created: {session_id} for user: {user_id}")
    return session_id


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session details"""
    return active_sessions.get(session_id)


def validate_session(session_id: str) -> Optional[str]:
    """Validate session and return user_id if valid"""
    session = active_sessions.get(session_id)
    
    if session is None:
        logger.warning(f"Session not found: {session_id}")
        return None
    
    if not session.get("active"):
        logger.warning(f"Session is inactive: {session_id}")
        return None
    
    # Update last activity
    session["last_activity"] = datetime.utcnow().isoformat()
    
    return session.get("user_id")


def invalidate_session(session_id: str) -> None:
    """Invalidate/logout session"""
    if session_id in active_sessions:
        active_sessions[session_id]["active"] = False
        logger.info(f"Session invalidated: {session_id}")


def invalidate_all_user_sessions(user_id: str) -> None:
    """Invalidate all sessions for a user (logout all devices)"""
    invalidated_count = 0
    for session_id, session_data in active_sessions.items():
        if session_data.get("user_id") == user_id:
            session_data["active"] = False
            invalidated_count += 1
    
    logger.info(f"Invalidated {invalidated_count} sessions for user: {user_id}")


def cleanup_old_sessions(max_age_hours: int = 24) -> int:
    """Remove old inactive sessions"""
    cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
    sessions_to_remove = []
    
    for session_id, session_data in active_sessions.items():
        if not session_data.get("active"):
            created_at = datetime.fromisoformat(session_data.get("created_at", ""))
            if created_at < cutoff_time:
                sessions_to_remove.append(session_id)
    
    for session_id in sessions_to_remove:
        del active_sessions[session_id]
    
    if sessions_to_remove:
        logger.info(f"Cleaned up {len(sessions_to_remove)} old sessions")
    
    return len(sessions_to_remove)


# ============================================================================
# AUTHENTICATION ROUTES
# ============================================================================

router = APIRouter(tags=["Authentication"])


@router.post("/register", response_model=UserRegisterResponse, status_code=status.HTTP_201_CREATED)
@log_exceptions
async def register(request: UserRegisterRequest):
    """Register a new user"""
    logger.info(f"Registration attempt for email: {request.email}")
    
    try:
        # Check if user already exists
        from db_manager import _db_manager
        existing_user = _db_manager.db[COLLECTION_USERS].find_one({
            "$or": [
                {"email": request.email},
                {"username": request.username}
            ]
        })
        
        if existing_user:
            if existing_user.get("email") == request.email:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already registered"
                )
            else:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already taken"
                )
        
        # Create new user
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(request.password)
        
        user_doc = {
            "user_id": user_id,
            "email": request.email,
            "username": request.username,
            "password_hash": hashed_password,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "last_login": None,
            "active": True
        }
        
        _db_manager.db[COLLECTION_USERS].insert_one(user_doc)
        logger.info(f"User registered successfully: {user_id}")
        
        return UserRegisterResponse(
            status="success",
            message="User registered successfully",
            user_id=user_id,
            email=request.email,
            username=request.username
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=UserLoginResponse)
@log_exceptions
async def login(request: UserLoginRequest):
    """Login user and create session"""
    logger.info(f"Login attempt for email: {request.email}")
    
    try:
        # Find user by email
        from db_manager import _db_manager
        user = _db_manager.db[COLLECTION_USERS].find_one({"email": request.email})
        
        if not user:
            logger.warning(f"Login failed: user not found for email {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        if not user.get("active"):
            logger.warning(f"Login failed: user account inactive {user.get('user_id')}")
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is inactive"
            )
        
        # Verify password
        if not verify_password(request.password, user.get("password_hash", "")):
            logger.warning(f"Login failed: invalid password for email {request.email}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Create tokens and session
        user_id = user.get("user_id")
        access_token, expires_in = create_access_token(user_id)
        refresh_token = create_refresh_token(user_id)
        session_id = create_session(user_id)
        
        # Update last login
        update_user_last_login(user_id)
        
        logger.info(f"User logged in successfully: {user_id}")
        
        return UserLoginResponse(
            status="success",
            message="Login successful",
            user_id=user_id,
            session_id=session_id,
            access_token=access_token,
            refresh_token=refresh_token,
            expires_in=expires_in
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )


@router.post("/refresh", response_model=TokenRefreshResponse)
@log_exceptions
async def refresh_token(request: TokenRefreshRequest):
    """Refresh access token using refresh token"""
    logger.info("Token refresh attempt")
    
    try:
        # Verify refresh token
        user_id = verify_token(request.refresh_token, token_type="refresh")
        
        if not user_id:
            logger.warning("Token refresh failed: invalid refresh token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        # Create new access token
        access_token, expires_in = create_access_token(user_id)
        
        logger.info(f"Token refreshed for user: {user_id}")
        
        return TokenRefreshResponse(
            status="success",
            access_token=access_token,
            expires_in=expires_in
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed"
        )


@router.post("/logout")
@log_exceptions
async def logout(session_id: str):
    """Logout user and invalidate session"""
    logger.info(f"Logout attempt for session: {session_id}")
    
    try:
        # Validate session exists
        user_id = validate_session(session_id)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        # Invalidate session
        invalidate_session(session_id)
        logger.info(f"User logged out: {user_id}")
        
        return {
            "status": "success",
            "message": "Logged out successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.post("/logout-all")
@log_exceptions
async def logout_all(session_id: str):
    """Logout from all devices (invalidate all user sessions)"""
    logger.info(f"Logout all attempt for session: {session_id}")
    
    try:
        # Validate session exists
        user_id = validate_session(session_id)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        # Invalidate all user sessions
        invalidate_all_user_sessions(user_id)
        logger.info(f"User logged out from all devices: {user_id}")
        
        return {
            "status": "success",
            "message": "Logged out from all devices successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Logout all error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed"
        )


@router.get("/profile", response_model=UserProfileResponse)
@log_exceptions
async def get_profile(session_id: str):
    """Get user profile information"""
    logger.info(f"Profile request for session: {session_id}")
    
    try:
        # Validate session
        user_id = validate_session(session_id)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        # Get user from database
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserProfileResponse(
            user_id=user.get("user_id"),
            email=user.get("email"),
            username=user.get("username"),
            created_at=user.get("created_at"),
            last_login=user.get("last_login"),
            active=user.get("active")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile retrieval error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve profile"
        )


@router.post("/verify-token")
@log_exceptions
async def verify_access_token(token: str):
    """Verify if access token is valid"""
    logger.debug("Token verification attempt")
    
    try:
        user_id = verify_token(token, token_type="access")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token"
            )
        
        return {
            "status": "valid",
            "user_id": user_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token verification error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token verification failed"
        )


@router.post("/change-password")
@log_exceptions
async def change_password(
    session_id: str,
    old_password: str,
    new_password: str,
    confirm_password: str
):
    """Change user password"""
    logger.info(f"Password change attempt for session: {session_id}")
    
    try:
        # Validate session
        user_id = validate_session(session_id)
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid session"
            )
        
        # Validate new passwords match
        if new_password != confirm_password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New passwords do not match"
            )
        
        # Validate password length
        if len(new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 8 characters"
            )
        
        # Get user and verify old password
        user = get_user_by_id(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        if not verify_password(old_password, user.get("password_hash", "")):
            logger.warning(f"Password change failed: invalid old password for {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid old password"
            )
        
        # Update password
        from db_manager import _db_manager
        hashed_new_password = hash_password(new_password)
        
        _db_manager.db[COLLECTION_USERS].update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "password_hash": hashed_new_password,
                    "updated_at": datetime.utcnow().isoformat()
                }
            }
        )
        
        logger.info(f"Password changed for user: {user_id}")
        
        return {
            "status": "success",
            "message": "Password changed successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Password change error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed"
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================

@router.get("/health")
async def auth_health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len([s for s in active_sessions.values() if s.get("active")])
    }


logger.info("auth.py loaded successfully")
