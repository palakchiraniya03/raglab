from typing import List, Tuple

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Tuple[str, int, int]]:
    """
    Splits text into chunks of maximum `chunk_size` characters, 
    with `chunk_overlap` characters overlapping between chunks.
    
    Returns a list of tuples: (chunk_text, char_start, char_end).
    """
    if not text:
        return []
    
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
        
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append((chunk, start, end))
        
        if end == text_length:
            break
            
        start += (chunk_size - chunk_overlap)
        
    return chunks
