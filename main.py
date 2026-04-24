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
    """Run a quality-check test with 10 nurses × 28 days (Feb)."""

    # Build test data
    nurses = [
        {
            "id": i, "name": f"看護師{i}", "position": "一般",
            "noNightShift": False, "noDayShift": False,
            "maxNightShifts": 6, "excludeFromGeneration": False,
        }
        for i in range(1, 11)
    ]
    config = {
        "weekdayDayStaff": 4,
        "weekendDayStaff": 3,
        "nightShiftPattern": [2, 2],
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
    # February 2026 weekends (0-indexed): Sat=6,Sun=0 → day indices
    weekends = [3, 4, 10, 11, 17, 18, 24, 25]

    test_input = {
        "nurses": nurses,
        "daysInMonth": 28,
        "year": 2026,
        "month": 1,  # 0-indexed: February
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

    errors = []
    data = p["data"]

    if not data:
        return {"status": "FAIL", "error": "No solution found", "metrics": p.get("metrics", {})}

    # Quality checks
    for nid, shifts in data.items():
        for d, s in enumerate(shifts):
            # Null check
            if s is None or s == "":
                errors.append(f"Nurse {nid} Day {d+1}: 空白セル")

            # 夜勤→明 check
            if s == "夜" and d + 1 < len(shifts) and shifts[d + 1] != "明":
                errors.append(f"Nurse {nid} Day {d+1}: 夜の翌日が明でない({shifts[d+1]})")

            # 管夜→管明 check
            if s == "管夜" and d + 1 < len(shifts) and shifts[d + 1] != "管明":
                errors.append(f"Nurse {nid} Day {d+1}: 管夜の翌日が管明でない({shifts[d+1]})")

            # 明→休or有 check
            if s == "明" and d + 1 < len(shifts) and shifts[d + 1] not in ("休", "有"):
                errors.append(f"Nurse {nid} Day {d+1}: 明の翌日が休or有でない({shifts[d+1]})")

            # 管明→休or有 check
            if s == "管明" and d + 1 < len(shifts) and shifts[d + 1] not in ("休", "有"):
                errors.append(f"Nurse {nid} Day {d+1}: 管明の翌日が休or有でない({shifts[d+1]})")

        # Consecutive working days check
        work_shifts = {"日", "夜", "管夜"}
        consec = 0
        max_consec_found = 0
        for d, s in enumerate(shifts):
            if s in work_shifts:
                consec += 1
                max_consec_found = max(max_consec_found, consec)
            else:
                consec = 0
        if max_consec_found > config["maxConsecutiveDays"]:
            errors.append(f"Nurse {nid}: 連続勤務{max_consec_found}日 (上限{config['maxConsecutiveDays']})")

        # 月末夜勤チェック
        if shifts[-1] in ("夜", "管夜"):
            errors.append(f"Nurse {nid}: 最終日に夜勤")
        if len(shifts) >= 2 and shifts[-2] in ("夜", "管夜"):
            errors.append(f"Nurse {nid}: 最終日-1に夜勤")

    # Daily staffing summary
    daily_summary = []
    for d in range(28):
        night_count = sum(1 for shifts in data.values() if shifts[d] == "夜")
        kan_night_count = sum(1 for shifts in data.values() if shifts[d] == "管夜")
        day_count = sum(1 for shifts in data.values() if shifts[d] == "日")
        daily_summary.append({
            "day": d + 1,
            "night": night_count,
            "kan_night": kan_night_count,
            "day_shift": day_count,
        })

    # Request fulfillment check
    req_checks = []
    for nid_str, reqs in requests_data.items():
        for day_str, req_type in reqs.items():
            day_idx = int(day_str) - 1
            actual = data.get(nid_str, [None] * 28)[day_idx] if nid_str in data else None
            expected = req_type
            matched = actual == expected
            req_checks.append({
                "nurse": nid_str, "day": int(day_str),
                "requested": expected, "actual": actual, "matched": matched,
            })

    # Night shift counts per nurse
    night_per_nurse = {}
    for nid, shifts in data.items():
        night_per_nurse[nid] = sum(1 for s in shifts if s == "夜")

    # NG pair check
    ng_violations = []
    for pair in night_ng_pairs:
        a, b = str(pair[0]), str(pair[1])
        if a in data and b in data:
            for d in range(28):
                if data[a][d] == "夜" and data[b][d] == "夜":
                    ng_violations.append(f"Day {d+1}: Nurse {a} and {b} both on night")

    return {
        "status": "OK" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors[:20],
        "score": p["score"],
        "metrics": p["metrics"],
        "daily_summary": daily_summary,
        "night_per_nurse": night_per_nurse,
        "request_checks": req_checks,
        "ng_violations": ng_violations,
    }
