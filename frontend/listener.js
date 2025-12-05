
/**
 * Context Heatmap Listener
 * Captures user attention signals (Selections & Copies) to improve RAG.
 */

const HEAT_CONFIG = {
    debounceMs: 800,       // Wait for user to finish selecting
    minChars: 15,          // Ignore clicking/fidgeting
    maxChars: 4000         // Ignore "Select All"
};

let selectionTimer = null;

document.addEventListener('selectionchange', () => {
    // 1. Debounce: Don't fire while mouse is dragging
    clearTimeout(selectionTimer);
    
    selectionTimer = setTimeout(() => {
        const selection = window.getSelection();
        const text = selection.toString().trim();

        // 2. Filter Noise
        if (text.length < HEAT_CONFIG.minChars) return;
        if (text.length > HEAT_CONFIG.maxChars) return;

        // 3. Locate Source Message (Assuming data-id attribute exists)
        // Adjust '.message-bubble' to match your UI's CSS class
        const messageNode = selection.anchorNode.parentElement.closest('.message-bubble');
        if (!messageNode) return;

        const messageId = messageNode.dataset.id;
        
        // 4. Capture Range (Relative to the message content)
        // Note: Real implementation needs strict offset calculation
        const payload = {
            type: 'implicit_highlight',
            message_id: messageId,
            text_snippet: text,
            timestamp: Date.now()
        };

        console.log("🔥 Heat Signal Detected:", payload);
        // sendToBackend(payload); 

    }, HEAT_CONFIG.debounceMs);
});

// 5. The "Gold" Signal: Explicit Copy
document.addEventListener('copy', () => {
    const selection = window.getSelection();
    // If a copy happens immediately after a selection, upgrade the heat score
    // Logic: Send a "upgrade_signal" event to backend for the last highlight
});
