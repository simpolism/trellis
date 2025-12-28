/**
 * Trellis HTMX Extensions
 * =======================
 *
 * Custom HTMX behaviors for Trellis UI:
 * - Button state management during SSE streaming
 * - Prompt flash animation
 * - Error handling
 */

// Button state management extension
htmx.defineExtension('sse-disable', {
    onEvent: function(name, evt) {
        // Disable buttons when SSE stream starts
        if (name === 'htmx:sseBeforeMessage' || name === 'htmx:sseOpen') {
            const buttons = document.querySelectorAll('.option-btn, button.reject-btn, button.skip-btn, button.undo-btn');
            buttons.forEach(btn => {
                btn.disabled = true;
                btn.classList.add('generating');
            });
        }

        // Re-enable buttons when stream completes
        if (name === 'htmx:sseMessage') {
            const data = evt.detail;
            if (data.event === 'complete' || data.event === 'done') {
                const buttons = document.querySelectorAll('.option-btn, button.reject-btn, button.skip-btn, button.undo-btn');
                buttons.forEach(btn => {
                    btn.disabled = false;
                    btn.classList.remove('generating');
                });
            }
        }

        // Handle SSE errors
        if (name === 'htmx:sseError') {
            console.error('SSE connection error:', evt.detail);
            const buttons = document.querySelectorAll('.option-btn, button.reject-btn, button.skip-btn, button.undo-btn');
            buttons.forEach(btn => {
                btn.disabled = false;
                btn.classList.remove('generating');
            });
        }
    }
});

// Prompt flash animation helper
function flashPrompt() {
    const promptCard = document.querySelector('.prompt-card');
    if (promptCard) {
        promptCard.classList.add('prompt-flash');
        setTimeout(() => {
            promptCard.classList.remove('prompt-flash');
        }, 600);
    }
}

// Inline prompt editing helpers
function toggleEdit() {
    const display = document.getElementById('prompt-display');
    const edit = document.getElementById('prompt-edit');
    const editButtons = document.getElementById('edit-buttons');
    const editBtn = document.getElementById('edit-btn');

    if (display && edit && editButtons && editBtn) {
        // Copy current prompt to textarea
        edit.value = display.textContent.replace('Prompt:\n\n', '').trim();

        // Toggle visibility
        display.style.display = 'none';
        edit.style.display = 'block';
        editButtons.style.display = 'flex';
        editBtn.style.display = 'none';
    }
}

function cancelEdit() {
    const display = document.getElementById('prompt-display');
    const edit = document.getElementById('prompt-edit');
    const editButtons = document.getElementById('edit-buttons');
    const editBtn = document.getElementById('edit-btn');

    if (display && edit && editButtons && editBtn) {
        display.style.display = 'block';
        edit.style.display = 'none';
        editButtons.style.display = 'none';
        editBtn.style.display = 'inline-block';
    }
}

// Accordion toggle
function toggleAccordion(id) {
    const content = document.getElementById(id);
    if (content) {
        content.style.display = content.style.display === 'none' ? 'block' : 'none';
    }
}

// Global error handler for HTMX
document.body.addEventListener('htmx:responseError', function(evt) {
    console.error('HTMX request failed:', evt.detail);
    alert('Request failed. Please try again.');
});

// Log SSE events for debugging
document.body.addEventListener('htmx:sseMessage', function(evt) {
    console.log('SSE event:', evt.detail.event, evt.detail.data);
});
