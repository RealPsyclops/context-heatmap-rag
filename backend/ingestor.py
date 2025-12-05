class SignalIngestor:
    def __init__(self, llm_client):
        self.llm = llm_client

    def process_signal(self, user_input_text: str, recent_signals: List[HeatSignal]):
        """
        Called when the user sends a message.
        Checks if the message validates or invalidates recent highlights.
        """
        for signal in recent_signals:
            # Check 1: Did they Paste it?
            if signal.text_snippet in user_input_text:
                # Check 2: Sentiment Analysis (The "Poison Pill")
                intent = self.llm.classify_intent(
                    user_text=user_input_text, 
                    quoted_text=signal.text_snippet
                )
                
                if intent == "CORRECTION":
                    print(f"🚫 POISON DETECTED. Ignoring signal for msg {signal.message_id}")
                    # Optionally: Add a 'Cooling' range to suppress future retrieval
                    continue
                
                elif intent == "VALIDATION":
                    print(f"✅ CONFIRMED. Boosting heat for msg {signal.message_id}")
                    self._apply_heat(signal)
