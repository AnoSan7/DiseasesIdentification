// Moved from inline <script> in index.html
document.addEventListener('DOMContentLoaded', () => {
    // Theme toggle: adds/removes `.dark` on <html> to switch to dark perspective
    const toggle = document.getElementById('dark-toggle');
    const root = document.documentElement;

    function setTheme(theme) {
        const isDark = theme === 'dark';
        if (isDark) root.classList.add('dark'); else root.classList.remove('dark');
        if (toggle) toggle.setAttribute('aria-checked', String(isDark));
        const knob = toggle && toggle.querySelector('.toggle-knob');
        if (knob) {
            if (isDark) knob.classList.add('toggled'); else knob.classList.remove('toggled');
        }
    }

    // Initialize from localStorage (default: light)
    try {
        const stored = localStorage.getItem('theme');
        if (stored) setTheme(stored);
        else setTheme('light');
    } catch (e) { setTheme('light'); }

    if (toggle) {
        toggle.addEventListener('click', () => {
            const isNowDark = !root.classList.contains('dark');
            setTheme(isNowDark ? 'dark' : 'light');
            try { localStorage.setItem('theme', isNowDark ? 'dark' : 'light'); } catch (e) {}
        });
        toggle.addEventListener('keydown', (e) => {
            if (e.key === ' ' || e.key === 'Spacebar' || e.key === 'Enter') {
                e.preventDefault();
                toggle.click();
            }
        });
    }

    // Form toggles
    const btnBlood = document.getElementById('btn-blood');
    const btnDiabetes = document.getElementById('btn-diabetes');
    const btnCardio = document.getElementById('btn-cardio');
    const btnLiver = document.getElementById('btn-liver');
    const formBlood = document.getElementById('form-blood');
    const formDiabetes = document.getElementById('form-diabetes');
    const formCardio = document.getElementById('form-cardio');
    const formLiver = document.getElementById('form-liver');
    const localOutput = document.getElementById('local-output');

    function showForm(name) {
        if (formBlood) formBlood.classList.add('hidden');
        if (formDiabetes) formDiabetes.classList.add('hidden');
        if (formCardio) formCardio.classList.add('hidden');
        if (formLiver) formLiver.classList.add('hidden');
        if (localOutput) localOutput.classList.add('hidden');
        if (name === 'blood' && formBlood) formBlood.classList.remove('hidden');
        if (name === 'diabetes' && formDiabetes) formDiabetes.classList.remove('hidden');
        if (name === 'cardio' && formCardio) formCardio.classList.remove('hidden');
        if (name === 'liver' && formLiver) formLiver.classList.remove('hidden');
    }

    if (btnBlood) btnBlood.addEventListener('click', () => showForm('blood'));
    if (btnDiabetes) btnDiabetes.addEventListener('click', () => showForm('diabetes'));
    if (btnCardio) btnCardio.addEventListener('click', () => showForm('cardio'));
    if (btnLiver) btnLiver.addEventListener('click', () => showForm('liver'));

    // Handle submits locally
    function formToObject(form) {
        const data = {};
        new FormData(form).forEach((v,k) => { data[k] = v; });
        return data;
    }

    function showLocalOutput(title, obj) {
        if (!localOutput) return;
        localOutput.innerHTML = '<strong>'+title+'</strong><pre class="mt-2 text-xs">'+JSON.stringify(obj, null, 2)+'</pre>';
        localOutput.classList.remove('hidden');
        // scroll into view
        localOutput.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function renderProbabilities(containerId, probabilities) {
        const container = document.getElementById(containerId + '-probs');
        if (!container) return;
        container.innerHTML = '';
        if (!probabilities) return;
        const list = document.createElement('ul');
        list.style.margin = '0';
        list.style.paddingLeft = '1rem';
        probabilities.forEach((p, idx) => {
            const li = document.createElement('li');
            li.innerText = `Class ${idx}: ${(Number(p) * 100).toFixed(1)}%`;
            list.appendChild(li);
        });
        container.appendChild(list);
    }

    // Add submit handlers
    // Blood form: send to backend model best_xgb_model2
    if (formBlood) {
        formBlood.addEventListener('submit', (e) => {
            e.preventDefault();
            const obj = formToObject(formBlood);
            const modelName = 'best_xgb_model2';

            postPredict(modelName, obj)
                .then(res => {
                    const resultEl = document.getElementById('blood-result');
                    if (resultEl) resultEl.textContent = String(res?.result ?? JSON.stringify(res));
                    renderProbabilities('blood-result', res?.probabilities);
                })
                .catch(err => {
                    const resultEl = document.getElementById('blood-result');
                    if (resultEl) resultEl.textContent = 'Prediction failed: ' + (err.message || String(err));
                    console.error('Prediction error', err);
                });

            const btn = formBlood.querySelector('button[type="submit"]');
            if (btn) {
                btn.classList.add('scale-95');
                setTimeout(() => btn.classList.remove('scale-95'), 150);
            }
        });
    }

    // Diabetes form: send to backend model best_xgb_model
    if (formDiabetes) {
        formDiabetes.addEventListener('submit', (e) => {
            e.preventDefault();
            const obj = formToObject(formDiabetes);
            const modelName = 'best_xgb_model';

            postPredict(modelName, obj)
                .then(res => {
                    const resultEl = document.getElementById('diabetes-result');
                    if (resultEl) resultEl.textContent = String(res?.result ?? JSON.stringify(res));
                    renderProbabilities('diabetes-result', res?.probabilities);
                })
                .catch(err => {
                    // show error message under form
                    const resultEl = document.getElementById('diabetes-result');
                    if (resultEl) resultEl.textContent = 'Prediction failed: ' + (err.message || String(err));
                    console.error('Prediction error', err);
                });

            const btn = formDiabetes.querySelector('button[type="submit"]');
            if (btn) {
                btn.classList.add('scale-95');
                setTimeout(() => btn.classList.remove('scale-95'), 150);
            }
        });
    }

    // POST to backend predict endpoint
    async function postPredict(modelName, payload) {
        try {
            const res = await fetch(`/predict/${modelName}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            if (!res.ok) {
                const text = await res.text();
                throw new Error(`Server error: ${res.status} ${text}`);
            }
            return await res.json();
        } catch (e) {
            throw e;
        }
    }

    // Cardio form submit -> best_xgb_model3
    if (formCardio) {
        formCardio.addEventListener('submit', (e) => {
            e.preventDefault();
            const raw = formToObject(formCardio);
            // coerce/massage fields the model expects
            const obj = Object.assign({}, raw);
            // gender codes: let's map Male->1, Female->0, Other->2 (adjust if your model used different mapping)
            if (typeof obj.gender === 'string') {
                const g = obj.gender.toLowerCase();
                if (g === 'male') obj.gender = 1;
                else if (g === 'female') obj.gender = 0;
                else obj.gender = 2;
            }
            // ensure binary fields are integers
            ['smoke','alco','active'].forEach(k => {
                if (k in obj) obj[k] = Number(obj[k]);
            });

            const modelName = 'best_xgb_model3';

            postPredict(modelName, obj)
                .then(res => {
                    const resultEl = document.getElementById('cardio-result');
                    if (resultEl) resultEl.textContent = String(res?.result ?? JSON.stringify(res));
                    renderProbabilities('cardio-result', res?.probabilities);
                })
                .catch(err => {
                    const resultEl = document.getElementById('cardio-result');
                    if (resultEl) resultEl.textContent = 'Prediction failed: ' + (err.message || String(err));
                    console.error('Prediction error', err);
                });

            const btn = formCardio.querySelector('button[type="submit"]');
            if (btn) {
                btn.classList.add('scale-95');
                setTimeout(() => btn.classList.remove('scale-95'), 150);
            }
        });
    }

    // Liver form submit -> best_xgb_model4
    if (formLiver) {
        formLiver.addEventListener('submit', (e) => {
            e.preventDefault();
            const raw = formToObject(formLiver);
            const obj = Object.assign({}, raw);
            // model expects Gender_Male binary feature
            if (typeof obj.Gender === 'string') {
                obj.Gender_Male = (obj.Gender.toLowerCase() === 'male') ? 1 : 0;
            } else {
                obj.Gender_Male = Number(obj.Gender) || 0;
            }
            // remove original Gender to avoid ambiguity
            delete obj.Gender;

            const modelName = 'best_xgb_model4';
            postPredict(modelName, obj)
                .then(res => {
                    const resultEl = document.getElementById('liver-result');
                    if (resultEl) resultEl.textContent = String(res?.result ?? JSON.stringify(res));
                    renderProbabilities('liver-result', res?.probabilities);
                })
                .catch(err => {
                    const resultEl = document.getElementById('liver-result');
                    if (resultEl) resultEl.textContent = 'Prediction failed: ' + (err.message || String(err));
                    console.error('Prediction error', err);
                });

            const btn = formLiver.querySelector('button[type="submit"]');
            if (btn) {
                btn.classList.add('scale-95');
                setTimeout(() => btn.classList.remove('scale-95'), 150);
            }
        });
    }

    // initialize with blood form visible
    showForm('blood');
});
