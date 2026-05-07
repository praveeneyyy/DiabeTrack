document.addEventListener('DOMContentLoaded', function () {
    function markSelections(form) {
        if (!form) return;
        form.querySelectorAll('.form-check').forEach(function (item) {
            const input = item.querySelector('input[type="radio"], input[type="checkbox"]');
            if (!input) return;
            item.classList.toggle('selected', input.checked);
        });
    }

    document.querySelectorAll('.form-check').forEach(function (item) {
        const input = item.querySelector('input[type="radio"], input[type="checkbox"]');
        if (!input) return;

        item.addEventListener('click', function (event) {
            if (event.target.tagName.toLowerCase() === 'label') return;
            input.checked = true;
            input.dispatchEvent(new Event('change', { bubbles: true }));
        });

        input.addEventListener('change', function () {
            markSelections(item.closest('form'));
        });
    });

    document.querySelectorAll('form').forEach(markSelections);

    document.querySelectorAll('textarea, input, select').forEach(function (control) {
        control.addEventListener('focus', function () {
            const card = control.closest('.card');
            if (card) card.classList.add('focus-ring');
        });
        control.addEventListener('blur', function () {
            const card = control.closest('.card');
            if (card) card.classList.remove('focus-ring');
        });
    });
});
