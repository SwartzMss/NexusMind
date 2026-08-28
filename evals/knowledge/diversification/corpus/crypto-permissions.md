# Crypto permissions and IPC

Crypto Crypto permission enforcement connects callers to the protected key service.
Binder Binder Binder Binder carries permission metadata across the Android service boundary.
Binder Binder records the caller identity before cryptographic work begins.
权限校验 权限校验 权限校验 权限校验 blocks unauthorized key operations at entry.
权限校验 权限校验 records an auditable decision before importing key material.
