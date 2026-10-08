# Sustituye a la configuración de `transformers` (ver birefnet.py).


class BiRefNetConfig:
    def __init__(self, bb_pretrained=False, **kwargs):
        self.bb_pretrained = bb_pretrained
