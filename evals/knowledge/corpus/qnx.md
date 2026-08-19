# QNX Neutrino

QNX Neutrino uses a microkernel architecture. Drivers, filesystems, and many services run as user-space processes instead of sharing one large kernel address space. Clients and servers coordinate through synchronous message passing, which makes service boundaries explicit.

Priority inheritance helps a server execute work on behalf of a higher-priority client without unnecessary priority inversion. A resource manager exposes device or service behavior through familiar pathname operations. The design emphasizes deterministic scheduling, fault containment, and small privileged components for embedded systems.
