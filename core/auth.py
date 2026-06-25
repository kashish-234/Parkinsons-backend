from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt
import requests
import time

from core.config import settings

security = HTTPBearer(auto_error=True)

JWKS_URL = f"{settings.supabase_url}/auth/v1/keys"

jwks_cache = None
jwks_last_fetched = 0


def get_jwks():
    global jwks_cache, jwks_last_fetched

    if jwks_cache is None or time.time() - jwks_last_fetched > 3600:
        res = requests.get(JWKS_URL)
        jwks_cache = res.json()
        jwks_last_fetched = time.time()

    return jwks_cache


def verify_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    token = credentials.credentials

    try:
        jwks = get_jwks()

        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header["kid"]

        key = next((k for k in jwks["keys"] if k["kid"] == kid), None)

        if key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token key"
            )

        issuer = f"{settings.supabase_url}/auth/v1"

        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience="authenticated",
            issuer=issuer
        )

        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("user_metadata", {}).get("role", "user")
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {str(e)}"
        )