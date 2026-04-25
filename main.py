import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from solver import solve_schedule

app = FastAPI(title="Nurse Schedule Solver API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY = os.environ.get("API_KEY", "kyoaikai-solver-2026")


def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/solve")
async def solve(request: Request):
    verify_api_key(request)
    body = await request.json()
    patterns = solve_schedule(body)
    return {"patterns": patterns}


@app.get("/test")
def test_solver():
    """品質チェック (24名 × 31日, 2026年3月、実運用に近い条件)。

    新solver仕様の検証:
      - 自動生成セル(=希望なし)に 管夜・管明・有 が出ない
      - 夜→明→休 / 管夜→管明→休 が100%守られる
      - 各日の夜勤人数 == nightShiftPattern (最終日のみ 0)
      - 各日の日勤人数 >= weekday/weekendDayStaff
      - 連続勤務上限を超えない
      - 空白セルなし
    """

    num_days = 31
    nurses = [
        {
            "id": i, "name": f"看護師{i}", "position": "一般",
            "noNightShift": i > 20, "noDayShift": False,
            "maxNightShifts": 6, "excludeFromGeneration": False,
        }
        for i in range(1, 25)
    ]
    config = {
        "weekdayDayStaff": 10,
        "weekendDayStaff": 6,
        "nightShiftPattern": [4, 4],
        "maxNightShifts": 6,
        "maxDaysOff": 10,
        "maxConsecutiveDays": 3,
        "maxDoubleNightPairs": 2,
        "excludeMgmtFromNightCount": False,
    }
    requests_data = {
        "1": {"5": "休", "10": "有"},
        "3": {"15": "休"},
    }
    night_ng_pairs = [[1, 2]]
    # 2026年3月の土日 (0-based)
    weekends = [0, 6, 7, 13, 14, 20, 21, 27, 28]

    test_input = {
        "nurses": nurses,
        "daysInMonth": num_days,
        "year": 2026,
        "month": 2,
        "config": config,
        "requests": requests_data,
        "nightNgPairs": night_ng_pairs,
        "prevMonthConstraints": {},
        "holidays": [],
        "weekends": weekends,
        "numPatterns": 1,
    }

    patterns = solve_schedule(test_input)
    p = patterns[0]
    data = p["data"]

    if not data:
        return {"status": "FAIL", "error": "No solution found", "metrics": p.get("metrics", {})}

    errors = []
    weekend_set = set(weekends)
    work_shifts = {"日", "夜", "管夜"}

    # 希望で許可されたラベル位置の集合（管夜/管明/有/明 が出てよいセル）
    allowed_special = {}  # (nid_str, day_idx) -> label
    for nid, reqs in requests_data.items():
        for day_str, label in reqs.items():
            allowed_special[(nid, int(day_str) - 1)] = label

    # 1. 各セル単位のチェック
    for nid, shifts in data.items():
        for d, s in enumerate(shifts):
            # 空白
            if s is None or s == "":
                errors.append(f"Nurse {nid} Day {d+1}: 空白セル")

            # 管夜・管明・有 は希望以外で出てはならない
            if s in ("管夜", "管明", "有"):
                if allowed_special.get((nid, d)) != s:
                    # 管明 は管夜の翌日に自動入る（管夜が希望にある場合）→ 許可
                    prev = shifts[d - 1] if d > 0 else None
                    if s == "管明" and prev == "管夜":
                        pass
                    else:
                        errors.append(f"Nurse {nid} Day {d+1}: 自動生成で {s} が出現")

            # 夜→明
            if s == "夜" and d + 1 < len(shifts) and shifts[d + 1] != "明":
                errors.append(f"Nurse {nid} Day {d+1}: 夜の翌日が明でない({shifts[d+1]})")
            # 管夜→管明
            if s == "管夜" and d + 1 < len(shifts) and shifts[d + 1] != "管明":
                errors.append(f"Nurse {nid} Day {d+1}: 管夜の翌日が管明でない({shifts[d+1]})")
            # 明→{休,夜,有}
            if s == "明" and d + 1 < len(shifts) and shifts[d + 1] not in ("休", "夜", "有"):
                errors.append(f"Nurse {nid} Day {d+1}: 明の翌日が休/夜/有以外({shifts[d+1]})")
            # 管明→{休,有}
            if s == "管明" and d + 1 < len(shifts) and shifts[d + 1] not in ("休", "有"):
                errors.append(f"Nurse {nid} Day {d+1}: 管明の翌日が休/有以外({shifts[d+1]})")

        # 連続勤務
        consec = 0
        max_consec_found = 0
        for s in shifts:
            if s in work_shifts:
                consec += 1
                max_consec_found = max(max_consec_found, consec)
            else:
                consec = 0
        if max_consec_found > config["maxConsecutiveDays"]:
            errors.append(f"Nurse {nid}: 連続勤務{max_consec_found}日 (上限{config['maxConsecutiveDays']})")

        # 月末夜勤 (最終日のみ禁止)
        if shifts[-1] in ("夜", "管夜"):
            errors.append(f"Nurse {nid}: 最終日に夜勤")

    # 2. 日次 staffing チェック (ハード制約と同じ条件)
    daily_summary = []
    for d in range(num_days):
        night_count = sum(1 for shifts in data.values() if shifts[d] == "夜")
        kan_night_count = sum(1 for shifts in data.values() if shifts[d] == "管夜")
        day_count = sum(1 for shifts in data.values() if shifts[d] == "日")

        # 最終日のみ夜勤禁止(=0)。それ以外は pattern=[4,4] で 4固定。
        expected_night = 0 if d == num_days - 1 else 4
        if night_count != expected_night:
            errors.append(f"Day {d+1}: 夜勤人数 {night_count} ≠ {expected_night}")

        req_day = config["weekendDayStaff"] if d in weekend_set else config["weekdayDayStaff"]
        if day_count < req_day:
            errors.append(f"Day {d+1}: 日勤人数 {day_count} < {req_day}")

        daily_summary.append({
            "day": d + 1,
            "night": night_count,
            "kan_night": kan_night_count,
            "day_shift": day_count,
        })

    # 3. 希望反映確認
    req_checks = []
    for nid_str, reqs in requests_data.items():
        for day_str, req_type in reqs.items():
            day_idx = int(day_str) - 1
            actual = data.get(nid_str, [None] * num_days)[day_idx] if nid_str in data else None
            matched = actual == req_type
            if not matched:
                errors.append(f"Nurse {nid_str} Day {day_str}: 希望 {req_type} が反映されてない (実際: {actual})")
            req_checks.append({
                "nurse": nid_str, "day": int(day_str),
                "requested": req_type, "actual": actual, "matched": matched,
            })

    # 4. 夜勤回数 (個人上限)
    night_per_nurse = {}
    for nid, shifts in data.items():
        cnt = sum(1 for s in shifts if s == "夜")
        night_per_nurse[nid] = cnt
        if cnt > config["maxNightShifts"]:
            errors.append(f"Nurse {nid}: 夜勤 {cnt}回 > 上限 {config['maxNightShifts']}")

    # 5. NG ペア
    ng_violations = []
    for pair in night_ng_pairs:
        a, b = str(pair[0]), str(pair[1])
        if a in data and b in data:
            for d in range(num_days):
                if data[a][d] == "夜" and data[b][d] == "夜":
                    ng_violations.append(f"Day {d+1}: Nurse {a} & {b} 同日夜勤")
                    errors.append(f"Day {d+1}: NGペア {a},{b} が同日夜勤")

    return {
        "status": "OK" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors[:30],
        "score": p["score"],
        "metrics": p["metrics"],
        "daily_summary": daily_summary,
        "night_per_nurse": night_per_nurse,
        "request_checks": req_checks,
        "ng_violations": ng_violations,
    }
