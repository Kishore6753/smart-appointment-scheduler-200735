"""
FastAPI backend for the Appointment Booking App.

Provides:
- User registration and login (JWT bearer auth)
- Appointment creation/listing/rescheduling/cancellation
- Admin availability management
- Basic notification hook stubs (server-side placeholders)

Environment variables (from .env in runtime):
- POSTGRES_URL, POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_PORT
- APPOINTMENT_APP_JWT_SECRET (required; set by orchestrator)
- APPOINTMENT_APP_JWT_EXPIRES_MINUTES (optional; default 720)
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import jwt
import psycopg2
import psycopg2.extras
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext

openapi_tags = [
    {"name": "Health", "description": "Service health checks."},
    {"name": "Auth", "description": "User registration and login."},
    {"name": "Appointments", "description": "Create and manage appointments."},
    {"name": "Availability", "description": "Admin availability management."},
    {"name": "Docs", "description": "Helpful usage notes."},
]

app = FastAPI(
    title="Appointment Booking API",
    description="Backend API for user auth, appointment booking, and admin availability.",
    version="0.1.0",
    openapi_tags=openapi_tags,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(dt: datetime) -> datetime:
    """Normalize datetimes to UTC."""
    if dt.tzinfo is None:
        # Assume naive datetimes are UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _db_conn() -> psycopg2.extensions.connection:
    """
    Create a new Postgres connection using runtime-provided env vars.

    We use individual env vars because the platform provides them via db_env_vars.
    """
    host = os.environ.get("POSTGRES_URL", "").strip()
    # Platform sometimes supplies POSTGRES_URL as 'postgresql://localhost:5000/myapp'
    # but the viewer server.js expects POSTGRES_URL for host name usage. We'll parse
    # conservatively: if it's a full URL, psycopg2 can accept it as dsn.
    user = os.environ.get("POSTGRES_USER")
    password = os.environ.get("POSTGRES_PASSWORD")
    dbname = os.environ.get("POSTGRES_DB")
    port = os.environ.get("POSTGRES_PORT")

    if not host:
        raise HTTPException(
            status_code=500,
            detail="Database not configured: missing POSTGRES_URL env var.",
        )

    try:
        if host.startswith("postgres://") or host.startswith("postgresql://"):
            # DSN style
            return psycopg2.connect(
                dsn=host,
                cursor_factory=psycopg2.extras.RealDictCursor,
            )
        # Host style
        return psycopg2.connect(
            host=host,
            user=user,
            password=password,
            dbname=dbname,
            port=int(port) if port else None,
            cursor_factory=psycopg2.extras.RealDictCursor,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {e}") from e


def _jwt_secret() -> str:
    secret = os.environ.get("APPOINTMENT_APP_JWT_SECRET")
    if not secret:
        # NOTE: Orchestrator should set this env var. We fail fast with a clear message.
        raise HTTPException(
            status_code=500,
            detail="Server auth not configured: missing APPOINTMENT_APP_JWT_SECRET env var.",
        )
    return secret


def _jwt_exp_minutes() -> int:
    val = os.environ.get("APPOINTMENT_APP_JWT_EXPIRES_MINUTES", "").strip()
    if not val:
        return 12 * 60
    try:
        return int(val)
    except ValueError:
        return 12 * 60


def _issue_token(user_id: str, role: str) -> str:
    exp = _utc_now() + timedelta(minutes=_jwt_exp_minutes())
    payload = {
        "sub": user_id,
        "role": role,
        "exp": exp,
        "iat": _utc_now(),
    }
    return jwt.encode(payload, _jwt_secret(), algorithm="HS256")


def _decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, _jwt_secret(), algorithms=["HS256"])
    except jwt.ExpiredSignatureError as e:
        raise HTTPException(status_code=401, detail="Token expired") from e
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token") from e


def _bearer_token(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")
    return parts[1].strip()


class AuthUser(BaseModel):
    id: str = Field(..., description="User id (uuid)")
    email: EmailStr = Field(..., description="User email")
    role: str = Field(..., description="User role: user|admin")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")


class RegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., min_length=8, description="User password (min 8 chars)")
    role: str = Field("user", description="Role to register as. Defaults to 'user'.")


class LoginRequest(BaseModel):
    email: EmailStr = Field(..., description="User email")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    user: AuthUser = Field(..., description="Authenticated user")





class AppointmentCreateRequest(BaseModel):
    start_time: datetime = Field(..., description="Appointment start time (ISO datetime)")
    end_time: datetime = Field(..., description="Appointment end time (ISO datetime)")
    notes: Optional[str] = Field(None, description="Optional notes")


class AppointmentRescheduleRequest(BaseModel):
    start_time: datetime = Field(..., description="New start time")
    end_time: datetime = Field(..., description="New end time")


class AppointmentOut(BaseModel):
    id: str = Field(..., description="Appointment id (uuid)")
    user_id: str = Field(..., description="Owner user id")
    start_time: datetime = Field(..., description="Start time (UTC)")
    end_time: datetime = Field(..., description="End time (UTC)")
    status: str = Field(..., description="scheduled|cancelled")
    notes: Optional[str] = Field(None, description="Notes")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Update timestamp (UTC)")


class AvailabilityUpsertRequest(BaseModel):
    start_time: datetime = Field(..., description="Available window start time")
    end_time: datetime = Field(..., description="Available window end time")
    is_available: bool = Field(True, description="Whether this window is available")


class AvailabilityOut(BaseModel):
    id: str = Field(..., description="Availability id (uuid)")
    admin_user_id: str = Field(..., description="Admin user id")
    start_time: datetime = Field(..., description="Start time (UTC)")
    end_time: datetime = Field(..., description="End time (UTC)")
    is_available: bool = Field(..., description="If available")
    created_at: datetime = Field(..., description="Creation timestamp (UTC)")
    updated_at: datetime = Field(..., description="Update timestamp (UTC)")


def _row_to_user(row: Dict[str, Any]) -> AuthUser:
    return AuthUser(
        id=str(row["id"]),
        email=row["email"],
        role=row["role"],
        created_at=row["created_at"],
    )


def _require_schema(conn: psycopg2.extensions.connection) -> None:
    """
    Ensure tables exist (idempotent).
    This keeps preview runnable even if init SQL wasn't applied yet.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_users (
              id UUID PRIMARY KEY,
              email TEXT UNIQUE NOT NULL,
              password_hash TEXT NOT NULL,
              role TEXT NOT NULL DEFAULT 'user',
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE TABLE IF NOT EXISTS availability_windows (
              id UUID PRIMARY KEY,
              admin_user_id UUID NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
              start_time TIMESTAMPTZ NOT NULL,
              end_time TIMESTAMPTZ NOT NULL,
              is_available BOOLEAN NOT NULL DEFAULT TRUE,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_avail_admin_time
              ON availability_windows (admin_user_id, start_time, end_time);

            CREATE TABLE IF NOT EXISTS appointments (
              id UUID PRIMARY KEY,
              user_id UUID NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,
              start_time TIMESTAMPTZ NOT NULL,
              end_time TIMESTAMPTZ NOT NULL,
              status TEXT NOT NULL DEFAULT 'scheduled',
              notes TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );

            CREATE INDEX IF NOT EXISTS idx_appt_user_time
              ON appointments (user_id, start_time, end_time);

            CREATE INDEX IF NOT EXISTS idx_appt_time
              ON appointments (start_time, end_time);
            """
        )
    conn.commit()


def _overlaps(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
    return a_start < b_end and b_start < a_end


def _notify(event: str, payload: Dict[str, Any]) -> None:
    """
    Notification hook stub.

    In the future, integrate email/SMS/webhook providers or Supabase triggers.
    For now, this is a no-op to keep a clear integration point.
    """
    # Intentionally minimal; replace with real integration later.
    _ = (event, payload)


def _get_current_user(authorization: Optional[str] = Header(default=None)) -> Tuple[str, str]:
    token = _bearer_token(authorization)
    decoded = _decode_token(token)
    return decoded["sub"], decoded.get("role", "user")


def _require_admin(user: Tuple[str, str] = Depends(_get_current_user)) -> Tuple[str, str]:
    user_id, role = user
    if role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user_id, role


@app.get("/", tags=["Health"], summary="Health check")
# PUBLIC_INTERFACE
def health_check():
    """Health check endpoint used by the platform preview."""
    return {"message": "Healthy"}


@app.get("/docs/help", tags=["Docs"], summary="API usage help")
# PUBLIC_INTERFACE
def docs_help():
    """Provides quick usage notes for auth and core routes."""
    return {
        "auth": {
            "register": "POST /auth/register {email,password,role?}",
            "login": "POST /auth/login {email,password} -> {access_token}",
            "header": "Authorization: Bearer <token>",
        },
        "appointments": {
            "create": "POST /appointments",
            "list_mine": "GET /appointments",
            "reschedule": "PATCH /appointments/{id}/reschedule",
            "cancel": "PATCH /appointments/{id}/cancel",
        },
        "availability_admin": {
            "upsert": "POST /admin/availability",
            "list": "GET /admin/availability",
            "delete": "DELETE /admin/availability/{id}",
        },
    }


@app.post(
    "/auth/register",
    tags=["Auth"],
    summary="Register a new user",
    response_model=TokenResponse,
)
# PUBLIC_INTERFACE
def register(req: RegisterRequest):
    """Register a user and return an auth token."""
    role = req.role if req.role in ("user", "admin") else "user"
    user_id = str(uuid.uuid4())
    password_hash = pwd_context.hash(req.password)

    conn = _db_conn()
    try:
        _require_schema(conn)
        with conn.cursor() as cur:
            try:
                cur.execute(
                    """
                    INSERT INTO app_users (id, email, password_hash, role)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id, email, role, created_at
                    """,
                    (user_id, req.email.lower(), password_hash, role),
                )
            except psycopg2.errors.UniqueViolation as e:
                conn.rollback()
                raise HTTPException(status_code=409, detail="Email already registered") from e

            row = cur.fetchone()
        conn.commit()

        token = _issue_token(str(row["id"]), row["role"])
        return TokenResponse(access_token=token, user=_row_to_user(row))
    finally:
        conn.close()


@app.post(
    "/auth/login",
    tags=["Auth"],
    summary="Login with email/password",
    response_model=TokenResponse,
)
# PUBLIC_INTERFACE
def login(req: LoginRequest):
    """Authenticate user and return an auth token."""
    conn = _db_conn()
    try:
        _require_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, email, role, created_at, password_hash
                FROM app_users
                WHERE email = %s
                """,
                (req.email.lower(),),
            )
            row = cur.fetchone()
        if not row or not pwd_context.verify(req.password, row["password_hash"]):
            raise HTTPException(status_code=401, detail="Invalid email or password")

        token = _issue_token(str(row["id"]), row["role"])
        user = AuthUser(id=str(row["id"]), email=row["email"], role=row["role"], created_at=row["created_at"])
        return TokenResponse(access_token=token, user=user)
    finally:
        conn.close()


@app.get(
    "/me",
    tags=["Auth"],
    summary="Get current user info",
    response_model=AuthUser,
)
# PUBLIC_INTERFACE
def me(user: Tuple[str, str] = Depends(_get_current_user)):
    """Return current authenticated user's info."""
    user_id, _role = user
    conn = _db_conn()
    try:
        _require_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, email, role, created_at FROM app_users WHERE id = %s",
                (user_id,),
            )
            row = cur.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="User not found")
        return _row_to_user(row)
    finally:
        conn.close()


def _validate_times(start_time: datetime, end_time: datetime) -> Tuple[datetime, datetime]:
    s = _parse_dt(start_time)
    e = _parse_dt(end_time)
    if e <= s:
        raise HTTPException(status_code=400, detail="end_time must be after start_time")
    if (e - s) > timedelta(hours=8):
        raise HTTPException(status_code=400, detail="Appointment duration too long (max 8h)")
    return s, e


def _check_availability_for_booking(
    conn: psycopg2.extensions.connection, start_time: datetime, end_time: datetime
) -> None:
    """
    Basic availability check:
    - There must exist at least one available window from any admin that fully covers the requested interval.
    - There must be no other scheduled appointment overlapping the interval.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM availability_windows
            WHERE is_available = TRUE
              AND start_time <= %s
              AND end_time >= %s
            """,
            (start_time, end_time),
        )
        ok = cur.fetchone()["cnt"] > 0
        if not ok:
            raise HTTPException(status_code=409, detail="No availability for requested time")

        cur.execute(
            """
            SELECT COUNT(*) AS cnt
            FROM appointments
            WHERE status = 'scheduled'
              AND start_time < %s
              AND end_time > %s
            """,
            (end_time, start_time),
        )
        if cur.fetchone()["cnt"] > 0:
            raise HTTPException(status_code=409, detail="Time slot already booked")


@app.post(
    "/appointments",
    tags=["Appointments"],
    summary="Create an appointment",
    response_model=AppointmentOut,
)
# PUBLIC_INTERFACE
def create_appointment(req: AppointmentCreateRequest, user: Tuple[str, str] = Depends(_get_current_user)):
    """Create an appointment for the authenticated user."""
    user_id, _role = user
    start_time, end_time = _validate_times(req.start_time, req.end_time)

    conn = _db_conn()
    try:
        _require_schema(conn)
        _check_availability_for_booking(conn, start_time, end_time)

        appt_id = str(uuid.uuid4())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO appointments (id, user_id, start_time, end_time, status, notes)
                VALUES (%s, %s, %s, %s, 'scheduled', %s)
                RETURNING id, user_id, start_time, end_time, status, notes, created_at, updated_at
                """,
                (appt_id, user_id, start_time, end_time, req.notes),
            )
            row = cur.fetchone()
        conn.commit()

        _notify("appointment.created", {"appointment_id": appt_id, "user_id": user_id})
        return AppointmentOut(**row)
    finally:
        conn.close()


@app.get(
    "/appointments",
    tags=["Appointments"],
    summary="List appointments (mine or all for admin)",
    response_model=List[AppointmentOut],
)
# PUBLIC_INTERFACE
def list_appointments(user: Tuple[str, str] = Depends(_get_current_user)):
    """List appointments for current user; admins see all."""
    user_id, role = user
    conn = _db_conn()
    try:
        _require_schema(conn)
        with conn.cursor() as cur:
            if role == "admin":
                cur.execute(
                    """
                    SELECT id, user_id, start_time, end_time, status, notes, created_at, updated_at
                    FROM appointments
                    ORDER BY start_time ASC
                    LIMIT 500
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT id, user_id, start_time, end_time, status, notes, created_at, updated_at
                    FROM appointments
                    WHERE user_id = %s
                    ORDER BY start_time ASC
                    LIMIT 500
                    """,
                    (user_id,),
                )
            rows = cur.fetchall()
        return [AppointmentOut(**r) for r in rows]
    finally:
        conn.close()


@app.patch(
    "/appointments/{appointment_id}/reschedule",
    tags=["Appointments"],
    summary="Reschedule an appointment",
    response_model=AppointmentOut,
)
# PUBLIC_INTERFACE
def reschedule_appointment(
    appointment_id: str,
    req: AppointmentRescheduleRequest,
    user: Tuple[str, str] = Depends(_get_current_user),
):
    """Reschedule an appointment owned by user (admins can reschedule any)."""
    user_id, role = user
    start_time, end_time = _validate_times(req.start_time, req.end_time)

    conn = _db_conn()
    try:
        _require_schema(conn)
        _check_availability_for_booking(conn, start_time, end_time)

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, status
                FROM appointments
                WHERE id = %s
                """,
                (appointment_id,),
            )
            current = cur.fetchone()
            if not current:
                raise HTTPException(status_code=404, detail="Appointment not found")
            if current["status"] != "scheduled":
                raise HTTPException(status_code=409, detail="Only scheduled appointments can be rescheduled")
            if role != "admin" and str(current["user_id"]) != user_id:
                raise HTTPException(status_code=403, detail="Not allowed")

            cur.execute(
                """
                UPDATE appointments
                SET start_time = %s,
                    end_time = %s,
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id, user_id, start_time, end_time, status, notes, created_at, updated_at
                """,
                (start_time, end_time, appointment_id),
            )
            row = cur.fetchone()
        conn.commit()

        _notify("appointment.rescheduled", {"appointment_id": appointment_id, "user_id": str(row["user_id"])})
        return AppointmentOut(**row)
    finally:
        conn.close()


@app.patch(
    "/appointments/{appointment_id}/cancel",
    tags=["Appointments"],
    summary="Cancel an appointment",
    response_model=AppointmentOut,
)
# PUBLIC_INTERFACE
def cancel_appointment(appointment_id: str, user: Tuple[str, str] = Depends(_get_current_user)):
    """Cancel an appointment owned by user (admins can cancel any)."""
    user_id, role = user
    conn = _db_conn()
    try:
        _require_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, status
                FROM appointments
                WHERE id = %s
                """,
                (appointment_id,),
            )
            current = cur.fetchone()
            if not current:
                raise HTTPException(status_code=404, detail="Appointment not found")
            if role != "admin" and str(current["user_id"]) != user_id:
                raise HTTPException(status_code=403, detail="Not allowed")
            if current["status"] == "cancelled":
                raise HTTPException(status_code=409, detail="Already cancelled")

            cur.execute(
                """
                UPDATE appointments
                SET status = 'cancelled',
                    updated_at = NOW()
                WHERE id = %s
                RETURNING id, user_id, start_time, end_time, status, notes, created_at, updated_at
                """,
                (appointment_id,),
            )
            row = cur.fetchone()
        conn.commit()

        _notify("appointment.cancelled", {"appointment_id": appointment_id, "user_id": str(row["user_id"])})
        return AppointmentOut(**row)
    finally:
        conn.close()


@app.post(
    "/admin/availability",
    tags=["Availability"],
    summary="Create/update an availability window (admin)",
    response_model=AvailabilityOut,
)
# PUBLIC_INTERFACE
def upsert_availability(req: AvailabilityUpsertRequest, admin: Tuple[str, str] = Depends(_require_admin)):
    """Admins can add availability windows used by booking checks."""
    admin_user_id, _ = admin
    start_time, end_time = _validate_times(req.start_time, req.end_time)

    conn = _db_conn()
    try:
        _require_schema(conn)

        # Simple approach: always insert new window
        avail_id = str(uuid.uuid4())
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO availability_windows (id, admin_user_id, start_time, end_time, is_available)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, admin_user_id, start_time, end_time, is_available, created_at, updated_at
                """,
                (avail_id, admin_user_id, start_time, end_time, req.is_available),
            )
            row = cur.fetchone()
        conn.commit()

        _notify("availability.updated", {"availability_id": avail_id, "admin_user_id": admin_user_id})
        return AvailabilityOut(**row)
    finally:
        conn.close()


@app.get(
    "/admin/availability",
    tags=["Availability"],
    summary="List availability windows (admin)",
    response_model=List[AvailabilityOut],
)
# PUBLIC_INTERFACE
def list_availability(admin: Tuple[str, str] = Depends(_require_admin)):
    """List all availability windows created by the current admin."""
    admin_user_id, _ = admin
    conn = _db_conn()
    try:
        _require_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, admin_user_id, start_time, end_time, is_available, created_at, updated_at
                FROM availability_windows
                WHERE admin_user_id = %s
                ORDER BY start_time ASC
                LIMIT 500
                """,
                (admin_user_id,),
            )
            rows = cur.fetchall()
        return [AvailabilityOut(**r) for r in rows]
    finally:
        conn.close()


@app.delete(
    "/admin/availability/{availability_id}",
    tags=["Availability"],
    summary="Delete availability window (admin)",
)
# PUBLIC_INTERFACE
def delete_availability(availability_id: str, admin: Tuple[str, str] = Depends(_require_admin)):
    """Delete an availability window."""
    admin_user_id, _ = admin
    conn = _db_conn()
    try:
        _require_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                """
                DELETE FROM availability_windows
                WHERE id = %s AND admin_user_id = %s
                """,
                (availability_id, admin_user_id),
            )
            if cur.rowcount == 0:
                raise HTTPException(status_code=404, detail="Availability window not found")
        conn.commit()
        _notify("availability.deleted", {"availability_id": availability_id, "admin_user_id": admin_user_id})
        return {"deleted": True}
    finally:
        conn.close()
