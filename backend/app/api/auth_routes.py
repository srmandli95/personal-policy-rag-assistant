from fastapi import APIRouter, Cookie, Depends, HTTPException, Response, status
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.auth.dependencies import get_current_user
from app.auth.jwt_handler import create_access_token
from app.auth.oauth_google import (
    GoogleOAuthError,
    build_google_authorization_url,
    create_oauth_state,
    exchange_google_code_for_token,
    fetch_google_userinfo,
    get_google_oauth_configured,
    validate_google_userinfo,
    verify_oauth_state,
)
from app.db.database import get_db
from app.models.user import User
from app.schemas.auth_schema import AuthUserResponse
from app.repositories.user_repository import get_or_create_oauth_user
from app.config.settings import settings
from app.utils.logger import get_logger


router = APIRouter(prefix="/auth", tags=["Auth"])
logger = get_logger(__name__)


def _mask_email(email: str | None) -> str:
    """Mask an email address before writing it to logs."""
    if not email or "@" not in email:
        return "missing"
    local_part, domain = email.split("@", 1)
    if not local_part:
        return f"***@{domain}"
    return f"{local_part[0]}***@{domain}"


def _build_access_token(user: User) -> str:
    """Create a signed access token for an authenticated user."""
    token_data = {
        "sub": user.id,
        "email": user.email,
        "auth_provider": user.auth_provider,
    }
    return create_access_token(data=token_data)


def _set_session_cookie(response: Response, access_token: str) -> None:
    """Attach the session token to the response as an HTTP-only cookie."""
    response.set_cookie(
        key=settings.AUTH_COOKIE_NAME,
        value=access_token,
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
        max_age=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        path="/",
    )
    logger.info(
        "Application session cookie set (cookie=%s, max_age_seconds=%s, secure=%s)",
        settings.AUTH_COOKIE_NAME,
        settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        settings.AUTH_COOKIE_SECURE,
    )


def _validate_google_oauth_state(state_value: str) -> None:
    """Validate and consume the Google OAuth state cookie."""
    if not verify_oauth_state(state_value):
        logger.warning("Google OAuth callback rejected: invalid or expired state")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired Google OAuth state",
        )
    logger.info("Google OAuth state validated")


def _require_google_oauth_configured() -> None:
    """Ensure Google OAuth settings are present before starting the flow."""
    if not get_google_oauth_configured():
        logger.warning("Google OAuth request rejected: configuration incomplete")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Google OAuth is not configured. Set GOOGLE_CLIENT_ID, "
                "GOOGLE_CLIENT_SECRET, and GOOGLE_REDIRECT_URI."
            ),
        )


@router.get("/me", response_model=AuthUserResponse)
def me(
    current_user: User = Depends(get_current_user),
) -> AuthUserResponse:
    """Return the authenticated user represented by the current session."""
    logger.info(
        "Authenticated user profile requested (user_id=%s, provider=%s)",
        current_user.id,
        current_user.auth_provider,
    )
    return AuthUserResponse.model_validate(current_user)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(response: Response) -> None:
    """Clear the browser session cookie for the current user."""
    response.delete_cookie(settings.AUTH_COOKIE_NAME, path="/")
    logger.info("Application session cookie cleared (cookie=%s)", settings.AUTH_COOKIE_NAME)


@router.get("/google/login")
def google_login(response: Response) -> dict[str, str]:
    """Start the Google OAuth login flow and redirect to Google."""
    logger.info("Google OAuth login started")
    _require_google_oauth_configured()

    try:
        state_value = create_oauth_state()
        authorization_url = build_google_authorization_url(
            state=state_value
        )
    except GoogleOAuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    response.set_cookie(
        key=settings.OAUTH_STATE_COOKIE_NAME,
        value=state_value,
        httponly=True,
        secure=settings.AUTH_COOKIE_SECURE,
        samesite="lax",
        max_age=600,
        path="/auth/google/callback",
    )
    logger.info(
        "Google OAuth authorization URL created and state cookie set "
        "(state_cookie=%s, state_max_age_seconds=%s, redirect_uri=%s)",
        settings.OAUTH_STATE_COOKIE_NAME,
        600,
        settings.GOOGLE_REDIRECT_URI,
    )
    return {"authorization_url": authorization_url}


@router.get("/google/callback")
async def google_callback(
    code: str | None = None,
    state: str | None = None,
    state_cookie: str | None = Cookie(default=None, alias=settings.OAUTH_STATE_COOKIE_NAME),
    db: Session = Depends(get_db),
) -> RedirectResponse:
    """Complete Google OAuth login and establish an application session."""
    logger.info(
        "Google OAuth callback received (has_code=%s, has_state=%s, has_state_cookie=%s)",
        bool(code),
        bool(state),
        bool(state_cookie),
    )
    if not code:
        logger.warning("Google OAuth callback rejected: missing authorization code")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google authorization code",
        )

    if not state:
        logger.warning("Google OAuth callback rejected: missing state")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Google OAuth state",
        )

    if not state_cookie or state != state_cookie:
        logger.warning(
            "Google OAuth callback rejected: state cookie missing or did not match"
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Google OAuth state did not match this browser session",
        )
    _validate_google_oauth_state(state)
    _require_google_oauth_configured()

    try:
        google_token = await exchange_google_code_for_token(code)
        google_access_token = google_token.get("access_token")
        if not google_access_token:
            raise GoogleOAuthError(
                "Google token response did not include an access token"
            )
        userinfo = await fetch_google_userinfo(str(google_access_token))
    except GoogleOAuthError as exc:
        logger.warning("Google OAuth provider step failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    try:
        identity = validate_google_userinfo(userinfo)
        logger.info(
            "Google OAuth identity validated (email=%s, provider_user_id=%s)",
            _mask_email(str(identity["email"])),
            identity["provider_user_id"],
        )
        user = await run_in_threadpool(
            get_or_create_oauth_user,
            db=db,
            provider="google",
            provider_user_id=str(identity["provider_user_id"]),
            email=str(identity["email"]),
            full_name=identity["full_name"],
        )
        logger.info(
            "Local OAuth user resolved (user_id=%s, email=%s)",
            user.id,
            _mask_email(user.email),
        )
    except ValueError as exc:
        logger.warning("Google OAuth identity rejected: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    access_token = _build_access_token(user)
    response = RedirectResponse(settings.FRONTEND_URL)
    _set_session_cookie(response, access_token)
    response.delete_cookie(
        settings.OAUTH_STATE_COOKIE_NAME,
        path="/auth/google/callback",
    )
    logger.info(
        "Google OAuth callback completed; redirecting to frontend (frontend_url=%s)",
        settings.FRONTEND_URL,
    )
    return response
