import numpy as np
from typing import List, Tuple
from .models import Message, Anchor

class ThermalRetriever:
    def __init__(self, messages: List[Message], anchors: List[Anchor]):
        self.messages = messages
        self.anchors = anchors
        
    def retrieve(self, query_embedding: List[float], top_k=5) -> List[Message]:
        """
        The Hybrid Retrieval Pipeline:
        1. Check Anchors (Direct Match)
        2. Check History (Vector Search + Heat Boost)
        """
        results = []
        
        # --- PHASE 1: The "Side Door" (Anchors) ---
        # Did the user ask for something they explicitly saved?
        best_anchor = self._search_anchors(query_embedding)
        if best_anchor:
            # If we found a strong anchor match (>0.85), inject it immediately
            target_msg = self._get_msg(best_anchor.source_message_id)
            results.append({
                "message": target_msg,
                "score": 2.0, # Artificial boost to ensure it's #1
                "reason": f"Matched User Anchor: '{best_anchor.user_label}'"
            })
            top_k -= 1 # We used one slot

        # --- PHASE 2: The "Thermal Search" ---
        scored_messages = []
        for msg in self.messages:
            # 1. Base Vector Score (Cosine Similarity)
            base_score = self._cosine_sim(query_embedding, msg.embedding)
            
            # 2. Calculate Heat Intensity
            # (What % of this message is covered by heat ranges?)
            heat_score = self._calculate_heat_density(msg)
            
            # 3. The Boost Formula
            # If heat is high, we forgive a lower vector match.
            # heat_score is 0.0 to 1.0. Alpha is the sensitivity.
            alpha = 0.5 
            final_score = base_score * (1 + (alpha * heat_score))
            
            scored_messages.append((msg, final_score))
            
        # Sort and take remaining top_k
        scored_messages.sort(key=lambda x: x[1], reverse=True)
        for msg, score in scored_messages[:top_k]:
            results.append({
                "message": msg,
                "score": score,
                "reason": "Vector Match + Heat Boost"
            })
            
        return results

    def _calculate_heat_density(self, msg: Message) -> float:
        """Returns 0.0 to 1.0 based on how much of the msg is 'hot'."""
        if not msg.heat_ranges:
            return 0.0
        
        # Simplified: Just sum lengths of ranges (ignoring overlaps for MVP)
        total_hot_chars = sum([end - start for start, end in msg.heat_ranges])
        density = total_hot_chars / len(msg.content)
        return min(density, 1.0) # Cap at 1.0

    def _search_anchors(self, query_vec) -> Optional[Anchor]:
        """Finds if the query matches a User Label."""
        best_score = -1
        best_anchor = None
        for anchor in self.anchors:
            score = self._cosine_sim(query_vec, anchor.label_embedding)
            if score > best_score:
                best_score = score
                best_anchor = anchor
        
        # Threshold: Only return if it's a very specific match
        return best_anchor if best_score > 0.85 else None

    def _get_msg(self, msg_id):
        return next((m for m in self.messages if m.id == msg_id), None)

    def _cosine_sim(self, v1, v2):
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
