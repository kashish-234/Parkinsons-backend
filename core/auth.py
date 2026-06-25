import time
import requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError

from core.config import settings

security = HTTPBearer(auto_error=True)

JWKS_URL = f"{settings.supabase_url}/auth/v1/keys"

_jwks_cache: dict | None = None
_jwks_last_fetched: float = 0
_JWKS_TTL = 3600  # 1 hour


def _get_jwks() -> dict:
    global _jwks_cache, _jwks_last_fetched

    if _jwks_cache is None or time.time() - _jwks_last_fetched > _JWKS_TTL:
        try:
            res = requests.get(JWKS_URL, timeout=10)
            res.raise_for_status()
            _jwks_cache = res.json()
            _jwks_last_fetched = time.time()
        except Exception as e:
            if _jwks_cache is not None:
                # Return stale cache rather than crashing
                import logging
                logging.getLogger(__name__).warning(
                    f"JWKS refresh failed ({e}); using stale cache."
                )
            else:
                raise RuntimeError(f"Cannot fetch JWKS from Supabase: {e}") from e

    return _jwks_cache


def verify_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    token = credentials.credentials

    try:
        jwks = _get_jwks()
        unverified_header = jwt.get_unverified_header(token)
        kid = unverified_header.get("kid")

        key = next(
            (k for k in jwks.get("keys", []) if k.get("kid") == kid),
            None,
        )

        if key is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: signing key not found",
            )

        issuer = f"{settings.supabase_url}/auth/v1"

        payload = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience="authenticated",
            issuer=issuer,
        )

        return {
            "user_id": payload.get("sub"),
            "email": payload.get("email"),
            "role": payload.get("user_metadata", {}).get("role", "user"),
        }

    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Auth error: {e}",
        )