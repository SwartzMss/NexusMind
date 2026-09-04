# QNX Resource Manager

## Pathname dispatch

QNX Resource Manager pathname dispatch maps an attached pathname to the
resource-manager connect and I/O functions registered for that namespace.
The dispatch layer provides the pathname boundary; the resource manager still
validates the request before performing device-specific work.

## Scheduling distractor

Thread scheduling priority affects when a server thread runs, but scheduling
does not replace pathname dispatch or authorize an I/O request.
