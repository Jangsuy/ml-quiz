const ANSWERS_KEY = 'ml-quiz-answers';
const COLLAPSED_KEY = 'ml-quiz-collapsed';
const TYPES = ['short', 'essay'];
const TYPE_LABEL = { short: '단답형', essay: '서술형' };

const listEl = document.getElementById('list');
const tabsEl = document.getElementById('type-tabs');
const jumpEl = document.getElementById('jump');

let activeType = typeFromHash();
const answers = readJSON(ANSWERS_KEY, {});
const collapsed = new Set(readJSON(COLLAPSED_KEY, []));

/* 저장 */

function readJSON(key, fallback) {
  try {
    const raw = localStorage.getItem(key);
    return raw === null ? fallback : JSON.parse(raw);
  } catch (err) {
    console.warn(`${key}를 읽지 못했습니다.`, err);
    return fallback;
  }
}

function writeJSON(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch (err) {
    console.warn(`${key}를 저장하지 못했습니다.`, err);
  }
}

function debounce(fn, ms) {
  let timer;
  return () => {
    clearTimeout(timer);
    timer = setTimeout(fn, ms);
  };
}

const persistAnswers = debounce(() => writeJSON(ANSWERS_KEY, answers), 300);

/* 수식 */

function renderMath(root) {
  if (typeof renderMathInElement !== 'function') return;
  renderMathInElement(root, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false },
      { left: '\\[', right: '\\]', display: true },
      { left: '\\(', right: '\\)', display: false },
    ],
    throwOnError: false,
  });
}

/* 크기 */

function autosize(el) {
  el.style.height = 'auto';
  el.style.height = el.scrollHeight + 'px';
}

// 숨겨진 요소는 scrollHeight가 0이라, 보이게 된 뒤에 다시 재야 한다.
function resizeVisible(root) {
  for (const el of root.querySelectorAll('.input')) {
    if (el.offsetParent !== null) autosize(el);
  }
}

function trackHeaderHeight() {
  const header = document.querySelector('.header');
  const apply = () => {
    document.documentElement.style.setProperty(
      '--header-h',
      header.offsetHeight + 'px'
    );
  };
  apply();
  new ResizeObserver(apply).observe(header);
}

/* 조립 */

function buildCard(q, number) {
  const card = document.createElement('article');
  card.className = 'card';

  const question = document.createElement('p');
  question.className = 'question';
  const num = document.createElement('span');
  num.className = 'num';
  num.textContent = 'Q' + number + '.';
  question.append(num, document.createTextNode(q.question));
  card.appendChild(question);

  if (q.tags.length) {
    const badges = document.createElement('div');
    badges.className = 'badges';
    for (const tag of q.tags) {
      const el = document.createElement('span');
      el.className = 'badge';
      el.textContent = '#' + tag;
      badges.appendChild(el);
    }
    card.appendChild(badges);
  }

  const input = document.createElement('textarea');
  input.className = 'input';
  input.rows = q.type === 'essay' ? 5 : 1;
  input.placeholder = '내 답 입력…';
  input.value = answers[q.id] || '';
  input.addEventListener('input', () => {
    answers[q.id] = input.value;
    autosize(input);
    persistAnswers();
  });
  card.appendChild(input);

  const answer = document.createElement('div');
  answer.className = 'answer';
  answer.textContent = q.answer;
  answer.hidden = true;

  const toggle = document.createElement('button');
  toggle.type = 'button';
  toggle.className = 'toggle';
  toggle.textContent = '정답 보기';
  toggle.setAttribute('aria-expanded', 'false');
  toggle.addEventListener('click', () => {
    const willOpen = answer.hidden;
    answer.hidden = !willOpen;
    toggle.textContent = willOpen ? '정답 숨기기' : '정답 보기';
    toggle.setAttribute('aria-expanded', String(willOpen));
  });

  card.append(toggle, answer);
  return card;
}

function buildSection(type, topic, items, startNumber) {
  const key = `${type}:${topic}`;
  const section = document.createElement('section');
  section.className = 'topic';
  section.id = 'topic-' + encodeURIComponent(key);

  const body = document.createElement('div');
  body.className = 'topic-body';

  const head = document.createElement('button');
  head.type = 'button';
  head.className = 'topic-head';
  const name = document.createElement('span');
  name.className = 'topic-name';
  name.textContent = topic;
  const count = document.createElement('span');
  count.className = 'topic-count';
  count.textContent = items.length;
  head.append(name, count);

  const setOpen = (open) => {
    body.hidden = !open;
    head.setAttribute('aria-expanded', String(open));
    if (open) resizeVisible(body);
  };

  head.addEventListener('click', () => {
    const open = body.hidden;
    if (open) collapsed.delete(key);
    else collapsed.add(key);
    writeJSON(COLLAPSED_KEY, [...collapsed]);
    setOpen(open);
  });

  items.forEach((q, i) => body.appendChild(buildCard(q, startNumber + i)));
  section.append(head, body);
  setOpen(!collapsed.has(key));

  return section;
}

function buildPane(type, questions) {
  const pane = document.createElement('div');
  pane.className = 'pane';
  pane.dataset.type = type;

  const items = questions.filter((q) => q.type === type);
  const topics = [...new Set(items.map((q) => q.topic))];

  const row = document.createElement('div');
  row.className = 'jump-row';
  row.dataset.type = type;

  let number = 1;
  for (const topic of topics) {
    const group = items.filter((q) => q.topic === topic);
    const section = buildSection(type, topic, group, number);
    number += group.length;
    pane.appendChild(section);

    const chip = document.createElement('button');
    chip.type = 'button';
    chip.className = 'chip';
    chip.textContent = topic;
    chip.addEventListener('click', () => {
      const body = section.querySelector('.topic-body');
      if (body.hidden) section.querySelector('.topic-head').click();
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
    row.appendChild(chip);
  }

  jumpEl.appendChild(row);
  return pane;
}

/* 유형 전환 */

function typeFromHash() {
  const hash = decodeURIComponent(location.hash.slice(1));
  return TYPES.includes(hash) ? hash : 'short';
}

function setType(type) {
  activeType = type;
  for (const el of document.querySelectorAll('.pane, .jump-row')) {
    el.hidden = el.dataset.type !== type;
  }
  for (const btn of tabsEl.querySelectorAll('.seg-btn')) {
    const on = btn.dataset.type === type;
    btn.classList.toggle('is-on', on);
    btn.setAttribute('aria-selected', String(on));
  }
  const pane = document.querySelector(`.pane[data-type="${type}"]`);
  if (pane) resizeVisible(pane);
}

tabsEl.addEventListener('click', (e) => {
  const btn = e.target.closest('.seg-btn');
  if (!btn) return;
  location.hash = btn.dataset.type;
  if (typeFromHash() === activeType) setType(activeType);
});

window.addEventListener('hashchange', () => setType(typeFromHash()));

/* 시작 */

function showError(message) {
  listEl.replaceChildren();
  const p = document.createElement('p');
  p.className = 'error';
  p.textContent = message;
  listEl.appendChild(p);
}

fetch('questions.json')
  .then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  })
  .then((questions) => {
    listEl.replaceChildren(...TYPES.map((t) => buildPane(t, questions)));
    renderMath(listEl);
    trackHeaderHeight();
    setType(activeType);
  })
  .catch((err) => {
    showError(
      `문제를 불러오지 못했습니다: ${err.message}\n` +
        'HTML 파일을 직접 연 경우라면 로컬 서버로 여세요.\n' +
        'python3 -m http.server 8000'
    );
  });
