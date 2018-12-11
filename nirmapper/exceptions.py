class WavefrontError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ReshapeError(Exception):
    def __init__(self, message):
        super().__init__(message)
