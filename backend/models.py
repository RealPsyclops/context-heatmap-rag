from dataclasses import dataclass, field
from typing import List, Optional
import time

@dataclass
class Message:
    id: str
    content: str
    role: str  # 'user' or 'assistant'
    embedding: List[float] = field(default_factory=list)
    # The Heatmap: A list of ranges [start, end] that are "Hot"
    heat_ranges: List[tuple] = field(default_factory=list)

@dataclass
class Anchor:
    """
    The 'Side Door'. Created when a user explicitly 'Saves' a message 
    and adds a label (e.g., 'My Login Logic').
    """
    id: str
    source_message_id: str
    user_label: str       # The search term (e.g. "Email Validator")
    label_embedding: List[float] # Embedding of the LABEL, not content
    created_at: float = field(default_factory=time.time)

@dataclass
class HeatSignal:
    message_id: str
    start_index: int
    end_index: int
    signal_type: str # 'hover', 'highlight', 'copy'
    
    @property
    def weight(self):
        if self.signal_type == 'copy': return 2.0
        if self.signal_type == 'highlight': return 1.0
        return 0.1
