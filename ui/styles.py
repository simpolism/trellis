"""
UI Styles
=========

Mobile-friendly CSS for the Trellis Gradio interface.
"""

MOBILE_CSS = """
/* Base responsive improvements */
.gradio-container {
    max-width: 100% !important;
}

/* Option groups - show full text with minimal padding */
.option-group {
    border: 1px solid var(--block-border-color);
    border-radius: 8px;
    padding: 8px 12px;
    margin-bottom: 8px;
    background: var(--background-fill-secondary);
}

.option-group:hover {
    border-color: var(--primary-500);
}

/* Option text - scrollable full content, minimal padding */
.option-text {
    max-height: 250px;
    overflow-y: auto;
    padding: 4px 0;
    font-size: 14px;
    line-height: 1.5;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* Remove extra margins from markdown inside options */
.option-text p {
    margin: 0 0 4px 0;
}

.option-text p:last-child {
    margin-bottom: 0;
}

/* Select button within option group */
.select-btn {
    margin-top: 4px;
}

/* Prompt display */
.prompt-card {
    background: var(--block-background-fill);
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid var(--block-border-color);
    margin-bottom: 8px;
}

/* Stats header */
.stats-header {
    display: flex;
    gap: 16px;
    padding: 8px 12px;
    background: var(--block-background-fill);
    border-radius: 6px;
    margin-bottom: 8px;
}

/* Mobile breakpoint */
@media (max-width: 768px) {
    .option-group {
        padding: 8px;
        margin-bottom: 6px;
    }

    .option-text {
        max-height: 180px;
        font-size: 13px;
    }

    /* Larger touch targets */
    button {
        min-height: 44px !important;
        padding: 12px 16px !important;
    }

    /* Stack action buttons */
    .action-row {
        flex-wrap: wrap !important;
        gap: 8px !important;
    }

    .action-row > * {
        flex: 1 1 45% !important;
        min-width: 120px !important;
    }

    /* Reduce padding on mobile */
    .gradio-container {
        padding: 8px !important;
    }

    /* Smaller headers */
    h1 { font-size: 1.5rem !important; }
    h2 { font-size: 1.25rem !important; }
    h3 { font-size: 1.1rem !important; }

    /* Full width inputs */
    input, textarea, select {
        width: 100% !important;
    }
}

/* Small mobile */
@media (max-width: 480px) {
    .option-text {
        max-height: 150px;
        font-size: 12px;
    }

    .action-row > * {
        flex: 1 1 100% !important;
    }

    /* Single column layout */
    .gr-row {
        flex-direction: column !important;
    }

    .gr-row > * {
        width: 100% !important;
        max-width: 100% !important;
    }
}

/* Section styling */
.section-group {
    border: 1px solid var(--block-border-color);
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
}

.section-title {
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--body-text-color);
}

/* Preview questions */
.preview-question {
    padding: 8px 12px;
    background: var(--background-fill-secondary);
    border-radius: 4px;
    margin: 4px 0;
    font-size: 13px;
}

/* VRAM display */
.vram-ok {
    color: var(--success-text-color, green);
}

.vram-warning {
    color: var(--error-text-color, red);
}

/* Journal display */
.journal-display {
    max-height: 500px;
    overflow-y: auto;
    padding: 12px;
    background: var(--background-fill-secondary);
    border-radius: 8px;
}

/* Disk usage warning */
.disk-warning {
    padding: 8px 12px;
    background: rgba(255, 193, 7, 0.1);
    border: 1px solid rgba(255, 193, 7, 0.3);
    border-radius: 4px;
    color: #856404;
    font-size: 13px;
}

/* Loading state for buttons */
.loading-btn {
    opacity: 0.7;
    pointer-events: none;
}

/* Smooth scroll behavior */
html {
    scroll-behavior: smooth;
}
"""
