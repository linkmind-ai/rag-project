class GistMemory:
    """
    Manages the accumulation and storage of proximal triples
    across multiple retrieval steps.
    """
    def __init__(self):
        self.proximal_triples = []
    
    def add_triples(self, new_triples):
        """Append new proximal triples to memory"""
        self.proximal_triples.extend(new_triples)
    
    def get_all_triples(self):
        """Return all accumulated triples"""
        return self.proximal_triples


def rrf_fusion(rank_lists, k = 60):
    scores = defaultdict(float)
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1 / (k + rank)

    return sorted(scores, key=scores.get, reverse=True)

def GeAR(query, max_steps=4):
    """
    GeAR pipeline implementation with multi-step retrieval capabilities.
    
    Parameters:
        query: Original input query  
        max_steps: Maximum number of retrieval steps
    """
    # Initialize variables
    gist_memory = GistMemory()
    current_query = query
    step = 1
    retrieved_passages = []
    
    while step <= max_steps:
        # Base retrieval for current query (i.e bm25)
        base_passages = base_retriever(current_query)
        
        # Read passages and extract proximal triples via LLM
        if step == 1:
            proximal_triples = reader(base_passages, query)
        else:
            proximal_triples = reader(base_passages, query, gist_memory.get_all_triples())
        
        # Link proximal triples to their closest real triples in index
        triples = tripleLink(proximal_triples)

        # Graph expansion using proximal triples
        expanded_passages = graph_expasion(triples, query)

        # Combine base and expanded passages, and save them
        combined_passages = rrf_fusion(base_passages + expanded_passages)
        retrieved_passages.append(combined_passages)
        
        # Read passages and extract proximal triples via LLM 
        proximal_triples = gist_memory_constructor(expanded_passages)

        # Add to gist memory
        gist_memory.add_triples(proximal_triples)

        # Check if we have enough evidence to answer query
        is_answerable, reasoning = reason(gist_memory.get_all_triples(), query)
        
        if is_answerable:
            break
        else:
            # Rewrite query for next step
            current_query = rewrite(query, gist_memory.get_all_triples(), reasoning)
            step += 1
    
    # Link final gist memory triples to passages
    gist_passages = []
    for triple in gist_memory.get_all_triples():
        linked_passages = passageLink(triple)
        gist_passages.append(linked_passages)
    
    # Final passage ranking combining all retrieved passages
    final_passages = rrf(gist_passages + retrieved_passages)

    return final_passages

def SyncGE(query):
    """
    SyncGE pipeline implementation.
    
    Parameters:
        query: Input query  
    """
    
    # Base retrieval for current query (i.e bm25)
    base_passages = base_retriever(current_query)
    
    # Read passages and extract proximal triples via LLM
    proximal_triples = reader(base_passages, query)
    
    # Link proximal triples to their closest real triples in index
    triples = tripleLink(proximal_triples)

    # Graph expansion using proximal triples
    expanded_passages = graph_expasion(triples, query)

    # Combine base and expanded passages, and save them
    combined_passages = rrf_fusion(base_passages + expanded_passages)

    return combined_passages

def diverse_triple_search(q, t_list, b, l, γ):
    """
    Performs diverse triple beam search, used in our proposed NaiveGE or SyncGE.

    Parameters:
        q: query
        b: beam size
        t_list: initial triples  
        l: maximum length
        γ: hyperparameter for diversity
    """
    # Initialize beam for first step
    B_0 = []

    # Score individual triples
    for t in t_list:
        s = score(q, [t])
        B_0.add((s, [t]))
    
    B_0 = top(B_0, b)  # Keep top b scoring triples
    
    # Iterative beam search
    for i in range(1, l):
        B = []
        
        for (s, T) in B_{i-1}:
            V = []  # Candidates from current path
            
            # Explore neighboring triples
            for t in get_neighbours(T[-1]):
                # Skip if triple already used
                if exists(t, B_{i-1}):
                    continue
                
                # Score new path with concatenated triple
                new_path = T + [t]
                s_new = s + score(q, new_path)
                V.add((s_new, new_path))
            
            sort(V, descending=True)
            
            # Apply diversity penalty
            for n in range(len(V)):
                s_new, path = V[n]
                penalty = exp(-min(n, γ)/γ)
                B.add((s_new * penalty, path))
        
        B_i = top(B, b)  # Keep top b paths
    
    return B_i