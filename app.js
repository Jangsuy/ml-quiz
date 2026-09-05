const ANSWERS_KEY = 'ml-quiz-answers';
const EXPANDED_KEY = 'ml-quiz-expanded';
const REJECTED_KEY = 'ml-quiz-rejected';
const HIDE_KEY = 'ml-quiz-hide-rejected';

const TYPES = ['short', 'essay'];

const listEl = document.getElementById('list');
const tabsEl = document.getElementById('type-tabs');
const jumpEl = document.getElementById('jump');
const statusEl = document.getElementById('status');
const statusCountEl = document.getElementById('status-count');
const copyBtn = document.getElementById('copy-btn');
const hideBtn = document.getElementById('hide-btn');

let activeType = typeFromHash();
const answers = readJSON(ANSWERS_KEY, {});
const expanded = new Set(readJSON(EXPANDED_KEY, []));
const rejected = new Set(readJSON(REJECTED_KEY, []));
let hideRejected = readJSON(HIDE_KEY, false) === true;

// 화면에 붙인 카드와 섹션. 별로 표시가 바뀔 때마다 여기를 훑어 갱신한다.
const cards = [];
const sections = [];

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

/* 별로 표시 */

function applyRejection() {
  for (const { q, el } of cards) {
    const off = rejected.has(q.id);
    el.classList.toggle('is-rejected', off);
    el.hidden = off && hideRejected;
  }

  for (const section of sections) {
    const left = section.items.filter(
      (it) => !(hideRejected && rejected.has(it.q.id))
    ).length;
    section.countEl.textContent = left;
    section.el.hidden = left === 0;
  }

  statusEl.hidden = rejected.size === 0;
  statusCountEl.textContent = `별로 표시 ${rejected.size}개`;
  hideBtn.textContent = hideRejected ? '보이기' : '숨기기';
  hideBtn.classList.toggle('is-on', hideRejected);
}

function rejectedIds() {
  return cards.filter(({ q }) => rejected.has(q.id)).map(({ q }) => q.id);
}

copyBtn.addEventListener('click', async () => {
  const text = rejectedIds().join(', ');
  try {
    await navigator.clipboard.writeText(text);
    copyBtn.textContent = '복사됨';
  } catch (err) {
    console.warn('클립보드 복사 실패', err);
    copyBtn.textContent = '복사 실패';
    window.prompt('아래 목록을 직접 복사하세요.', text);
  }
  setTimeout(() => {
    copyBtn.textContent = '목록 복사';
  }, 1500);
});

hideBtn.addEventListener('click', () => {
  hideRejected = !hideRejected;
  writeJSON(HIDE_KEY, hideRejected);
  applyRejection();
  const pane = document.querySelector(`.pane[data-type="${activeType}"]`);
  if (pane) resizeVisible(pane);
});

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

  const reject = document.createElement('button');
  reject.type = 'button';
  reject.className = 'reject';
  reject.textContent = '👎';
  reject.title = '별로인 문제로 표시';
  reject.setAttribute('aria-label', '별로인 문제로 표시');
  reject.setAttribute('aria-pressed', String(rejected.has(q.id)));
  reject.addEventListener('click', () => {
    if (rejected.has(q.id)) rejected.delete(q.id);
    else rejected.add(q.id);
    reject.setAttribute('aria-pressed', String(rejected.has(q.id)));
    writeJSON(REJECTED_KEY, [...rejected]);
    applyRejection();
  });

  const actions = document.createElement('div');
  actions.className = 'actions';
  actions.append(toggle, reject);

  card.append(actions, answer);
  return card;
}

function buildSection(type, topic, items, startNumber) {
  const key = `${type}:${topic}`;
  const section = document.createElement('section');
  section.className = 'topic';

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
    if (open) expanded.add(key);
    else expanded.delete(key);
    writeJSON(EXPANDED_KEY, [...expanded]);
    setOpen(open);
  });

  const entries = items.map((q, i) => {
    const el = buildCard(q, startNumber + i);
    body.appendChild(el);
    const entry = { q, el };
    cards.push(entry);
    return entry;
  });

  section.append(head, body);
  setOpen(expanded.has(key));

  sections.push({ el: section, countEl: count, items: entries, setOpen, head, body });
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
    applyRejection();
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
