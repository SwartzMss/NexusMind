# Secure Platform Security

## TrustZone

TrustZone keeps secure-world key operations isolated from normal-world
applications. A normal-world caller can request a service, but it cannot read
the secure-world key material directly.

## PKI

Certificate-chain validation checks signatures, issuer linkage, validity
intervals, and the trust anchor before a public key is accepted.
Possessing a certificate file alone does not prove that its chain is trusted.
