# ML Quiz

머신러닝 단답형·서술형 문제를 풀고 정답을 토글해서 확인하는 개인용 복습 페이지.

## 문제 추가

`questions.json`에 항목을 하나 더 넣고 push하면 1분 안에 반영된다.

```json
{
  "id": "q021",
  "type": "short",
  "tags": ["평가지표"],
  "question": "문제 내용",
  "answer": "정답. 줄바꿈은 \\n으로 넣으면 그대로 보인다."
}
```

- `id`는 겹치지 않게. 입력한 답을 브라우저에 저장하는 키라서 바꾸면 그 답이 사라진다.
- `type`은 `short` 또는 `essay`.
- `tags`는 상단 필터 칩으로 자동 반영된다.

## 로컬에서 보기

`questions.json`을 fetch하므로 파일을 더블클릭하면 안 열린다.

```
python3 -m http.server 8000
```

## 메모

입력한 답은 `localStorage`에 저장되므로 기기마다 따로 남는다.
