"""
UI Styles
=========

Brutalist late-80s/early-90s aesthetic for the Trellis Gradio interface.
Inspired by classic software design with dark green "trellis" accents.
"""

MOBILE_CSS = """
/* Import IBM Plex Mono for that classic terminal feel */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ========== Base Reset & Theme ========== */
:root {
    --trellis-green: #1a4d1a;
    --trellis-green-light: #2d6b2d;
    --trellis-green-pale: #e8f0e8;
    --cream: #f5f5f0;
    --white: #ffffff;
    --black: #1a1a1a;
    --gray-dark: #3a3a3a;
    --gray-mid: #888888;
    --gray-light: #cccccc;
    --gray-pale: #e8e8e8;
    --shadow: #a0a0a0;
    --danger: #8b0000;
}

/* Override Gradio's default font */
.gradio-container,
.gradio-container *,
.gr-button,
.gr-input,
.gr-box,
.prose {
    font-family: "IBM Plex Mono", "Courier New", Courier, monospace !important;
}

.gradio-container {
    max-width: 100% !important;
    background-color: var(--cream) !important;
}

/* Main app container */
.main,
.contain {
    background-color: var(--cream) !important;
}

/* ========== Typography ========== */
h1, h2, h3, h4, h5, h6 {
    font-family: "IBM Plex Mono", monospace !important;
    font-weight: 600 !important;
    color: var(--black) !important;
}

/* App title */
.gradio-container > .main h1:first-of-type {
    text-align: center;
    border-bottom: 2px solid var(--black);
    padding-bottom: 8px;
    margin-bottom: 4px;
}

/* Tagline */
.gradio-container > .main > div > p:first-of-type {
    text-align: center;
    font-style: italic;
    color: var(--gray-dark);
    margin-bottom: 16px;
}

/* ========== Tabs ========== */
.tabs {
    border: 2px solid var(--black) !important;
    background-color: var(--white) !important;
    box-shadow: 4px 4px 0 var(--shadow) !important;
}

.tab-nav {
    background-color: var(--gray-pale) !important;
    border-bottom: 2px solid var(--black) !important;
    gap: 0 !important;
}

.tab-nav button {
    border: none !important;
    border-right: 1px solid var(--black) !important;
    border-radius: 0 !important;
    background-color: var(--gray-light) !important;
    color: var(--black) !important;
    font-weight: 500 !important;
    padding: 10px 20px !important;
    transition: none !important;
}

.tab-nav button:last-child {
    border-right: none !important;
}

.tab-nav button.selected {
    background-color: var(--white) !important;
    border-bottom: 2px solid var(--white) !important;
    margin-bottom: -2px !important;
    font-weight: 600 !important;
}

.tab-nav button:hover:not(.selected) {
    background-color: var(--gray-pale) !important;
}

.tabitem {
    background-color: var(--white) !important;
    padding: 16px !important;
}

/* ========== Buttons ========== */
button,
.gr-button {
    font-family: "IBM Plex Mono", monospace !important;
    font-weight: 500 !important;
    border-radius: 0 !important;
    transition: none !important;
    cursor: pointer;
}

/* Default button - outset 3D style */
button.secondary,
button:not(.primary):not(.stop),
.gr-button-secondary {
    background-color: var(--gray-light) !important;
    color: var(--black) !important;
    border: 2px outset var(--gray-light) !important;
    box-shadow: none !important;
}

button.secondary:hover,
button:not(.primary):not(.stop):hover {
    background-color: var(--gray-pale) !important;
}

button.secondary:active,
button:not(.primary):not(.stop):active {
    border-style: inset !important;
}

/* Primary button - green */
button.primary,
.gr-button-primary {
    background-color: var(--trellis-green) !important;
    color: var(--white) !important;
    border: 2px outset var(--trellis-green-light) !important;
    box-shadow: none !important;
}

button.primary:hover {
    background-color: var(--trellis-green-light) !important;
}

button.primary:active {
    border-style: inset !important;
}

/* Stop/danger button */
button.stop {
    background-color: var(--danger) !important;
    color: var(--white) !important;
    border: 2px outset #a03030 !important;
}

button.stop:active {
    border-style: inset !important;
}

/* Disabled buttons */
button:disabled {
    opacity: 0.5 !important;
    cursor: not-allowed !important;
}

/* ========== Form Inputs ========== */
input,
textarea,
select,
.gr-input,
.gr-textarea {
    font-family: "IBM Plex Mono", monospace !important;
    background-color: var(--white) !important;
    border: 1px solid var(--black) !important;
    border-radius: 0 !important;
    padding: 6px 8px !important;
}

input:focus,
textarea:focus,
select:focus {
    outline: 2px solid var(--trellis-green) !important;
    outline-offset: -2px;
}

/* Slider */
input[type="range"] {
    accent-color: var(--trellis-green) !important;
}

/* Labels */
label,
.gr-input-label {
    font-weight: 500 !important;
    color: var(--black) !important;
    font-size: 13px !important;
}

/* Info text */
.gr-input-info,
span.info {
    color: var(--gray-dark) !important;
    font-size: 12px !important;
    font-style: italic;
}

/* ========== Groups & Accordions ========== */
.gr-group,
.gr-box,
.group {
    border: 1px solid var(--black) !important;
    border-radius: 0 !important;
    background-color: var(--white) !important;
    box-shadow: 2px 2px 0 var(--shadow) !important;
    padding: 12px !important;
}

/* Accordion headers */
.gr-accordion > .label-wrap {
    background-color: var(--gray-light) !important;
    border: 1px solid var(--black) !important;
    border-radius: 0 !important;
    padding: 8px 12px !important;
}

.gr-accordion > .label-wrap:hover {
    background-color: var(--gray-pale) !important;
}

/* ========== Stats Header ========== */
.stats-header {
    display: flex;
    gap: 16px;
    padding: 8px 12px;
    background-color: var(--trellis-green-pale) !important;
    border: 1px solid var(--trellis-green) !important;
    margin-bottom: 12px;
}

.stats-header p {
    margin: 0 !important;
    color: var(--black) !important;
}

/* Prevent loading opacity on stats - keep readable */
.stats-header,
.stats-header.pending,
.stats-header * {
    opacity: 1 !important;
    transition: none !important;
}

/* ========== Prompt Card ========== */
.prompt-card {
    background-color: var(--white) !important;
    padding: 12px 16px;
    border: 2px solid var(--black) !important;
    margin-bottom: 12px;
    box-shadow: 3px 3px 0 var(--shadow) !important;
}

.prompt-card p {
    margin: 0 !important;
}

/* Prevent loading opacity on prompt - keep readable */
.prompt-card,
.prompt-card.pending,
.prompt-card * {
    opacity: 1 !important;
    transition: none !important;
}

/* ========== Option Groups ========== */
.option-group {
    border: 1px solid var(--black) !important;
    padding: 8px 12px 10px 12px;
    margin-bottom: 8px;
    background-color: var(--white) !important;
    display: flex;
    flex-direction: column;
    gap: 8px;
    box-shadow: 2px 2px 0 var(--shadow) !important;
    border-radius: 0 !important;
}

.option-group:hover {
    border-color: var(--trellis-green) !important;
    border-width: 2px !important;
    padding: 7px 11px 9px 11px; /* Compensate for border width */
}

/* Option label */
.option-label {
    margin: 0 0 4px 0 !important;
}

.option-label p {
    margin: 0 !important;
    font-size: 14px;
    color: var(--trellis-green);
}

/* Option text - plain textbox display */
.option-text {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.option-text textarea {
    font-family: "IBM Plex Mono", monospace !important;
    font-size: 13px !important;
    line-height: 1.5 !important;
    background-color: var(--gray-pale) !important;
    border: 1px solid var(--gray-light) !important;
    padding: 8px !important;
    resize: none !important;
    cursor: default !important;
}

/* Select button */
.select-btn {
    align-self: flex-start;
}

/* ========== Action Row ========== */
.action-row {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.action-row button {
    flex: 1;
    min-width: 100px;
}

/* ========== Section Groups (Config) ========== */
.section-group {
    border: 1px solid var(--black) !important;
    padding: 12px;
    margin-bottom: 12px;
    background-color: var(--gray-pale) !important;
}

.section-title {
    font-weight: 600;
    margin-bottom: 8px;
    color: var(--black);
}

/* ========== Preview Questions ========== */
.preview-question {
    padding: 6px 10px;
    background-color: var(--white);
    border: 1px solid var(--gray-light);
    margin: 4px 0;
    font-size: 12px;
}

/* ========== VRAM Display ========== */
.vram-ok {
    color: var(--trellis-green) !important;
    font-weight: 600;
}

.vram-warning {
    color: var(--danger) !important;
    font-weight: 600;
}

/* ========== Journal Display ========== */
.journal-display {
    max-height: 450px;
    overflow-y: auto;
    padding: 12px;
    background-color: var(--white);
    border: 1px solid var(--black) !important;
}

/* Flatten nested divs */
.journal-display * {
    max-height: none !important;
    overflow: visible !important;
}

.journal-display {
    overflow-y: auto !important;
}

/* Prevent loading opacity */
.journal-display,
.journal-display.pending {
    opacity: 1 !important;
    transition: none !important;
}

/* ========== Disk Warning ========== */
.disk-warning {
    padding: 8px 12px;
    background-color: #fff8dc;
    border: 1px solid #b8860b;
    font-size: 13px;
    color: #654321;
}

/* ========== Markdown Styling ========== */
.prose {
    color: var(--black) !important;
}

.prose h1, .prose h2, .prose h3 {
    border-bottom: 1px solid var(--gray-light);
    padding-bottom: 4px;
}

.prose code {
    background-color: var(--gray-pale) !important;
    border: 1px solid var(--gray-light);
    padding: 1px 4px;
    font-size: 12px;
}

.prose pre {
    background-color: var(--gray-pale) !important;
    border: 1px solid var(--black);
    padding: 10px;
    box-shadow: 2px 2px 0 var(--shadow);
}

.prose hr {
    border: none;
    border-top: 1px solid var(--black);
    margin: 16px 0;
}

/* Horizontal rules in app */
.gradio-container hr {
    border: none !important;
    border-top: 1px solid var(--gray-light) !important;
    margin: 12px 0 !important;
}

/* ========== JSON Display ========== */
.json-holder {
    background-color: var(--white) !important;
    border: 1px solid var(--black) !important;
    border-radius: 0 !important;
}

/* ========== Dropdown ========== */
.gr-dropdown {
    border-radius: 0 !important;
}

/* ========== Loading State Override ========== */
.pending {
    opacity: 1 !important;
}

/* Gradio's internal loading indicators */
.wrap.svelte-1sk3mf9 {
    opacity: 1 !important;
}

/* ========== Scrollbar Styling (Retro) ========== */
::-webkit-scrollbar {
    width: 14px;
    height: 14px;
}

::-webkit-scrollbar-track {
    background: var(--gray-pale);
    border: 1px solid var(--gray-light);
}

::-webkit-scrollbar-thumb {
    background: var(--gray-light);
    border: 1px solid var(--gray-mid);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--gray-mid);
}

::-webkit-scrollbar-corner {
    background: var(--gray-pale);
}

/* ========== Mobile Responsive ========== */
@media (max-width: 768px) {
    .gradio-container {
        padding: 8px !important;
    }

    .tabs {
        box-shadow: 3px 3px 0 var(--shadow) !important;
    }

    .tab-nav button {
        padding: 8px 12px !important;
        font-size: 13px !important;
    }

    .tabitem {
        padding: 12px !important;
    }

    .option-group {
        padding: 6px 10px 8px 10px;
        margin-bottom: 6px;
    }

    .option-text {
        max-height: 160px;
        font-size: 12px;
    }

    .prompt-card {
        padding: 10px 12px;
        box-shadow: 2px 2px 0 var(--shadow) !important;
    }

    .stats-header {
        flex-wrap: wrap;
        gap: 8px;
        padding: 6px 10px;
        font-size: 12px;
    }

    /* Larger touch targets */
    button {
        min-height: 42px !important;
        padding: 10px 14px !important;
    }

    .action-row {
        gap: 6px;
    }

    .action-row button {
        min-width: 80px;
    }

    /* Typography */
    h1 { font-size: 1.4rem !important; }
    h2 { font-size: 1.2rem !important; }
    h3 { font-size: 1rem !important; }

    input, textarea, select {
        font-size: 16px !important; /* Prevent zoom on iOS */
    }
}

/* Small mobile */
@media (max-width: 480px) {
    .option-text {
        max-height: 140px;
        font-size: 11px;
    }

    .action-row {
        flex-direction: column;
    }

    .action-row button {
        width: 100%;
    }

    .stats-header {
        flex-direction: column;
        gap: 4px;
    }

    .tab-nav {
        flex-wrap: wrap;
    }

    .tab-nav button {
        flex: 1;
        min-width: 80px;
        padding: 8px !important;
        font-size: 12px !important;
    }
}
"""
