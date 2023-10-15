from jsonargparse import CLI
from paired.aistpp import AISTPP


def aistpp(root: str, stride: float = 0.5, length: int = 5):
    AISTPP.build(root=root, stride=stride, length=length)


if __name__ == "__main__":
    CLI(
        [
            aistpp,
        ],
        as_positional=False,
    )
