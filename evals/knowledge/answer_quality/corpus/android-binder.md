# Android Binder

## Oneway calls

An oneway Binder call can report a zero PID because it is processed asynchronously.
The zero PID describes the asynchronous transaction context rather than proving
that the caller is anonymous.

## Authorization

The kernel still supplies caller credentials for authorization.
SELinux policy evaluates the credential-bearing Binder transaction against the
target service policy. A valid Binder handle identifies an endpoint, not an
authorization decision.

## Review caveat

A zero PID always proves an anonymous caller is an unsupported conclusion.
The transaction's credentials and the service's policy remain the relevant
authorization evidence.
