"""Custom exceptions for Razorpay Test Mode integration."""

from __future__ import annotations


class RazorpayIntegrationError(Exception):
    """Base exception for all Razorpay adapter errors."""

    def __init__(self, message: str, status_code: int | None = None, error_code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code or "ERR_RAZORPAY_INTEGRATION"


class RazorpayAuthenticationError(RazorpayIntegrationError):
    """Raised when Razorpay credentials are invalid or missing."""

    def __init__(self, message: str = "Razorpay authentication failed or credentials missing.") -> None:
        super().__init__(message, status_code=401, error_code="ERR_RAZORPAY_AUTH")


class RazorpayNetworkError(RazorpayIntegrationError):
    """Raised when network connectivity to Razorpay fails."""

    def __init__(self, message: str = "Failed to connect to Razorpay API.") -> None:
        super().__init__(message, status_code=503, error_code="ERR_RAZORPAY_NETWORK")


class RazorpayTimeoutError(RazorpayIntegrationError):
    """Raised when Razorpay request exceeds timeout limit."""

    def __init__(self, message: str = "Razorpay API request timed out.") -> None:
        super().__init__(message, status_code=504, error_code="ERR_RAZORPAY_TIMEOUT")


class RazorpayAPIError(RazorpayIntegrationError):
    """Raised when Razorpay API returns a 4xx or 5xx response."""

    def __init__(self, message: str, status_code: int = 400, error_code: str = "ERR_RAZORPAY_API") -> None:
        super().__init__(message, status_code=status_code, error_code=error_code)


class RazorpayInvalidKeyModeError(RazorpayIntegrationError):
    """Raised when configured Razorpay key is not a Test Mode key (must start with 'rzp_test_')."""

    def __init__(
        self,
        message: str = "STRICT SAFETY VIOLATION: Only Razorpay Test Mode keys (rzp_test_*) are permitted. Execution blocked.",
    ) -> None:
        super().__init__(message, status_code=400, error_code="ERR_RAZORPAY_INVALID_KEY_MODE")

