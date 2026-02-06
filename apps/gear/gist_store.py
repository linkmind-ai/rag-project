from typing import Any


class GistMemory:
    """proximal triples 저장"""

    def __init__(self) -> None:
        self.proximal_triples: list[dict[str, Any]] = []

    def add_triples(self, new_triples: list[dict[str, Any]]) -> None:
        self.proximal_triples.extend(new_triples)

    def get_all_triples(self) -> list[dict[str, Any]]:

        return self.proximal_triples

    def clear(self) -> None:
        self.proximal_triples = []
