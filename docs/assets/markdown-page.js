/* global marked, renderMathInElement */

(function () {
    const DEFAULT_MARKED_OPTIONS = {
        gfm: true,
        breaks: false,
        headerIds: true,
        mangle: false,
    };

    async function loadMarkdownInto(container) {
        const mdPath = container.getAttribute('data-md');
        if (!mdPath) return;

        container.innerHTML = '<p class="muted">Loading workflow…</p>';

        let response;
        try {
            response = await fetch(mdPath, { cache: 'no-cache' });
        } catch (err) {
            container.innerHTML = '<p class="muted">Failed to load workflow.</p>';
            return;
        }

        if (!response.ok) {
            container.innerHTML = `<p class="muted">Workflow not found: ${mdPath}</p>`;
            return;
        }

        const markdown = await response.text();

        if (typeof marked === 'undefined') {
            container.textContent = markdown;
            return;
        }

        marked.setOptions(DEFAULT_MARKED_OPTIONS);
        container.innerHTML = marked.parse(markdown);

        if (typeof renderMathInElement === 'function') {
            try {
                renderMathInElement(container, {
                    delimiters: [
                        { left: '$$', right: '$$', display: true },
                        { left: '$', right: '$', display: false },
                    ],
                });
            } catch (_) {
                // ignore math render errors
            }
        }
    }

    document.addEventListener('DOMContentLoaded', function () {
        const container = document.querySelector('[data-md]');
        if (container) loadMarkdownInto(container);
    });
})();
