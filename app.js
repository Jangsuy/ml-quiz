const STORAGE_KEY = 'ml-quiz-answers';
const TYPE_LABEL = { short: '단답형', essay: '서술형' };

const listEl = document.getElementById('list');
const countEl = document.getElementById('count');
const typeFilterEl = document.getElementById('type-filter');
const tagFilterEl = document.getElementById('tag-filter');

let questions = [];
let activeType = 'all';
const activeTags = new Set();
let answers = loadAnswers();

/* 저장 */

function loadAnswers() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY)) || {};
  } catch (err) {
    console.warn('저장된 답을 읽지 못했습니다.', err);
    return {};
  }
}

function saveAnswers() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(answers));
  } catch (err) {
    console.warn('답을 저장하지 못했습니다.', err);
  }
}

function debounce(fn, ms) {
  let timer;
  return () => {
    clearTimeout(timer);
    timer = setTimeout(fn, ms);
  };
}

const persist = debounce(saveAnswers, 300);

/* 필터 */

function matches(q) {
  if (activeType !== 'all' && q.type !== activeType) return false;
  if (activeTags.size && !q.tags.some((t) => activeTags.has(t))) return false;
  return true;
}

function buildTagFilter() {
  const tags = [...new Set(questions.flatMap((q) => q.tags))].sort((a, b) =>
    a.localeCompare(b, 'ko')
  );
  for (const tag of tags) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'tag-btn';
    btn.textContent = '#' + tag;
    btn.addEventListener('click', () => {
      if (activeTags.has(tag)) activeTags.delete(tag);
      else activeTags.add(tag);
      btn.classList.toggle('is-on', activeTags.has(tag));
      render();
    });
    tagFilterEl.appendChild(btn);
  }
}

typeFilterEl.addEventListener('click', (e) => {
  const btn = e.target.closest('.seg-btn');
  if (!btn) return;
  activeType = btn.dataset.type;
  for (const b of typeFilterEl.querySelectorAll('.seg-btn')) {
    b.classList.toggle('is-on', b === btn);
  }
  render();
});

/* 렌더 */

function autosize(el) {
  el.style.height = 'auto';
  el.style.height = el.scrollHeight + 'px';
}

function buildCard(q, index) {
  const card = document.createElement('article');
  card.className = 'card';

  const badges = document.createElement('div');
  badges.className = 'badges';
  const typeBadge = document.createElement('span');
  typeBadge.className = 'badge type';
  typeBadge.textContent = TYPE_LABEL[q.type] || q.type;
  badges.appendChild(typeBadge);
  for (const tag of q.tags) {
    const el = document.createElement('span');
    el.className = 'badge';
    el.textContent = '#' + tag;
    badges.appendChild(el);
  }
  card.appendChild(badges);

  const question = document.createElement('p');
  question.className = 'question';
  const num = document.createElement('span');
  num.className = 'num';
  num.textContent = 'Q' + (index + 1) + '.';
  question.appendChild(num);
  question.appendChild(document.createTextNode(q.question));
  card.appendChild(question);

  const input = document.createElement('textarea');
  input.className = 'input';
  input.rows = q.type === 'essay' ? 5 : 1;
  input.placeholder = '내 답 입력…';
  input.value = answers[q.id] || '';
  input.addEventListener('input', () => {
    answers[q.id] = input.value;
    autosize(input);
    persist();
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

  card.appendChild(toggle);
  card.appendChild(answer);

  // 카드가 화면에 붙은 뒤라야 scrollHeight가 나온다.
  queueMicrotask(() => autosize(input));

  return card;
}

function render() {
  const shown = questions.filter(matches);
  listEl.replaceChildren();

  if (!shown.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = '조건에 맞는 문제가 없습니다.';
    listEl.appendChild(p);
  } else {
    shown.forEach((q, i) => listEl.appendChild(buildCard(q, i)));
  }

  countEl.textContent = `${shown.length} / ${questions.length}문항`;
}

function showError(message) {
  listEl.replaceChildren();
  countEl.textContent = '';
  const p = document.createElement('p');
  p.className = 'error';
  p.textContent = message;
  listEl.appendChild(p);
}

/* 시작 */

fetch('questions.json')
  .then((res) => {
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return res.json();
  })
  .then((data) => {
    questions = data;
    buildTagFilter();
    render();
  })
  .catch((err) => {
    showError(
      `문제를 불러오지 못했습니다: ${err.message}\n` +
        'HTML 파일을 직접 연 경우라면 로컬 서버로 여세요.\n' +
        'python3 -m http.server 8000'
    );
  });
