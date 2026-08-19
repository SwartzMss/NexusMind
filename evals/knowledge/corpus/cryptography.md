# Applied Cryptography

AES GCM provides authenticated encryption: it protects confidentiality and detects unauthorized modification. Reusing a nonce with the same key can catastrophically weaken security, so nonce construction and key lifecycle rules are part of the protocol rather than optional details.

HKDF performs key derivation from input key material and context. Ed25519 is a digital signature scheme used to authenticate messages or artifacts. Encryption, hashing, key derivation, and signatures solve different problems; secure systems choose explicit algorithms and bind identities, versions, and context into verified data.
