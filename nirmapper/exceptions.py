class WavefrontError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ColladaError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ReshapeError(Exception):
    def __init__(self, message):
        super().__init__(message)


class ModelError(Exception):
    def __init__(self, message):
        super().__init__(message)


class TextureError(Exception):
    def __init__(self, message):
        super().__init__(message)


class RenderError(Exception):
    def __init__(self, message):
        super().__init__(message)
