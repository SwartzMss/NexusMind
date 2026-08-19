# Arm TrustZone

TrustZone divides a system into a secure world and a normal world. A secure monitor coordinates transitions while hardware access controls isolate memory and peripherals. Trusted firmware in the secure world can protect keys and verify sensitive operations without trusting the normal operating system.

On newer Arm profiles, exception levels describe privilege while the security state remains a separate axis. A trusted execution environment uses this isolation boundary for small security services. TrustZone is not a cryptographic algorithm and does not replace careful firmware design, authenticated boot, or narrow interfaces.
