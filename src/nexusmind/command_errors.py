class CommandError(Exception):
    pass


class CommandConfigError(CommandError):
    pass


class CommandProfileError(CommandError):
    pass


class CommandStartError(CommandError):
    pass


class CommandLimitError(CommandError):
    pass


class CommandCleanupError(CommandError):
    pass
