import logging
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from core.config import settings

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=True)


def verify_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Verify a Supabase JWT by calling Supabase's /auth/v1/user endpoint.
    Supabase validates the token (any algorithm) and returns the user.
    No local key management needed — works with HS256, ES256, or any future algo.
    """
    token = credentials.credentials

    try:
        res = requests.get(
            f"{settings.supabase_url}/auth/v1/user",
            headers={
                "Authorization": f"Bearer {token}",
                "apikey": settings.supabase_anon_key,
            },
            timeout=10,
        )
    except requests.RequestException as e:
        logger.error("Supabase auth request failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Auth service unavailable",
        )

    if res.status_code != 200:
        logger.warning("Supabase rejected token: %s %s", res.status_code, res.text[:200])
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )

    user = res.json()
    return {
        "user_id": user.get("id"),
        "email": user.get("email"),
        "role": user.get("user_metadata", {}).get("role", "user"),
    }
