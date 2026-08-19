# Android Runtime and IPC

Android applications commonly communicate through Binder IPC. Binder carries typed transactions between processes while the framework's ActivityManager coordinates component lifecycle. Zygote preloads common runtime state and forks application processes, and ART executes managed application code.

Each application normally receives a distinct Linux identity and application sandbox. SELinux policy adds mandatory access control between applications, system services, and hardware-facing domains. These controls complement permission checks; they do not make every Binder interface safe automatically.
