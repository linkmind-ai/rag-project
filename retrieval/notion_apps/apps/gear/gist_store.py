from typing import List, Dict, Any


class GistMemory:
    """proximal triples 저장"""
    def __init__(self):
        self.proximal_triples: List[Dict[str, Any]] = []

    def add_triples(self, new_triples: List[Dict[str, Any]]) -> None:
        self.proximal_triples.extend(new_triples)

    def get_all_triples(self) -> List[Dict[str, Any]]:
        
        return self.proximal_triples
    
    def clear(self) -> None:
        self.proximal_triples = []