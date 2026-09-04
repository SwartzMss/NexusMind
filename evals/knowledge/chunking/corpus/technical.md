# Android Security

This guide explains the security boundaries used by Android system services.

## Binder

Binder provides the interprocess communication boundary for system services.

### Transaction

The oneway transaction uses pid zero because the recipient does not synchronously return a caller process.
