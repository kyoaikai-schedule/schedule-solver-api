# Nurse Schedule Solver API

OR-Tools CP-SATソルバーを使った看護師勤務表最適化API。

## セットアップ

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## API仕様

### ヘルスチェック

```
GET /health
```

レスポンス: `{"status": "ok"}`

### 勤務表生成

```
POST /solve
X-API-Key: kyoaikai-solver-2026
Content-Type: application/json
```

リクエストボディ:

```json
{
  "nurses": [
    {
      "id": 1,
      "name": "田中",
      "position": "一般",
      "noNightShift": false,
      "noDayShift": false,
      "maxNightShifts": 6,
      "excludeFromGeneration": false
    }
  ],
  "daysInMonth": 31,
  "year": 2026,
  "month": 4,
  "config": {
    "weekdayDayStaff": 6,
    "weekendDayStaff": 5,
    "nightShiftPattern": [4, 4],
    "maxNightShifts": 6,
    "maxDaysOff": 10,
    "maxConsecutiveDays": 3,
    "maxDoubleNightPairs": 2,
    "excludeMgmtFromNightCount": false
  },
  "requests": { "1": { "5": "休", "10": "有" } },
  "nightNgPairs": [[1, 2]],
  "prevMonthConstraints": { "1": { "_consecDays": 2 } },
  "holidays": [3, 29],
  "weekends": [0, 5, 6, 7, 12, 13, 14, 19, 20, 21, 26, 27, 28],
  "numPatterns": 3
}
```

レスポンス:

```json
{
  "patterns": [
    {
      "label": "パターンA",
      "data": { "1": ["日", "夜", "明", "休", "..."] },
      "score": 9500,
      "metrics": {
        "nightBalance": 0.5,
        "dayShortage": 0,
        "nightShortage": 0,
        "consecViolations": 0,
        "requestMatch": 85.0,
        "avgDaysOff": 10.2
      }
    }
  ]
}
```

### シフト種別

| コード | 意味 |
|--------|------|
| 日 | 日勤 |
| 夜 | 夜勤 |
| 明 | 明け |
| 管夜 | 管理夜勤 |
| 管明 | 管理明け |
| 休 | 休み |

## デプロイ (Render.com)

1. GitHubリポジトリを接続
2. `render.yaml` が自動検出される
3. 環境変数 `API_KEY` を設定
4. デプロイ実行
