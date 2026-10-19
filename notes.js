const noteEl = document.getElementById('note');
const tocEl = document.getElementById('toc');

// marked가 수식 안의 _, *, \\ 를 마크다운으로 해석하지 않도록
// 수식을 자리표시자로 바꿔 두었다가 HTML이 나온 뒤 KaTeX 결과로 되돌린다.
function protectMath(src) {
  const math = [];
  const stash = (tex, display) => `%%MATH${math.push({ tex, display }) - 1}%%`;
  const text = src
    .split(/(```[\s\S]*?```)/g)
    .map((part, i) => {
      if (i % 2 === 1) return part; // 코드 블록은 그대로 둔다
      return part
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, tex) => stash(tex, true))
        .replace(/\$([^$\n]+?)\$/g, (_, tex) => stash(tex, false));
    })
    .join('');
  return { text, math };
}

function restoreMath(html, math) {
  return html.replace(/%%MATH(\d+)%%/g, (_, i) => {
    const { tex, display } = math[Number(i)];
    return katex.renderToString(tex, { displayMode: display, throwOnError: false, strict: false });
  });
}

function buildToc() {
  noteEl.querySelectorAll('h1').forEach((h, i) => {
    h.id = `ch-${i + 1}`;
    const chip = document.createElement('a');
    chip.className = 'chip';
    chip.href = `#${h.id}`;
    chip.textContent = h.textContent;
    tocEl.append(chip);
  });
}

async function main() {
  try {
    const res = await fetch('notes.md', { cache: 'no-cache' });
    if (!res.ok) throw new Error(`notes.md를 불러오지 못했습니다. (HTTP ${res.status})`);
    const { text, math } = protectMath(await res.text());
    noteEl.innerHTML = restoreMath(marked.parse(text), math);
    noteEl.querySelectorAll('pre code').forEach((el) => hljs.highlightElement(el));
    buildToc();
    if (location.hash) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  } catch (err) {
    console.error(err);
    noteEl.innerHTML = '';
    const p = document.createElement('p');
    p.className = 'error';
    p.textContent = err.message;
    noteEl.append(p);
  }
}

main();
