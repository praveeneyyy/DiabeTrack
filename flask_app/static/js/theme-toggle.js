const themeToggle = document.getElementById('themeToggle');
const storedTheme = localStorage.getItem('diabetrack-theme') || 'light';

const applyTheme = theme => {
    document.body.classList.remove('theme-light', 'theme-dark');
    document.body.classList.add(`theme-${theme}`);
    if (themeToggle) {
        themeToggle.textContent = theme === 'dark' ? '☀' : '☾';
        themeToggle.title = theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode';
    }
    localStorage.setItem('diabetrack-theme', theme);
};

if (themeToggle) {
    themeToggle.addEventListener('click', () => {
        const nextTheme = document.body.classList.contains('theme-dark') ? 'light' : 'dark';
        applyTheme(nextTheme);
    });
}
applyTheme(storedTheme);
