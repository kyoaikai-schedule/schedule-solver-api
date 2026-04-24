"""OR-Tools CP-SAT nurse schedule solver — v2 full rewrite."""

from __future__ import annotations
from datetime import date
from ortools.sat.python import cp_model

# ──────────────────────────────────────
# Shift constants (0 is NOT used)
# ──────────────────────────────────────
DAY = 1        # 日勤
NIGHT = 2      # 夜勤
AKE = 3        # 明け
KAN_NIGHT = 4  # 管夜
KAN_AKE = 5    # 管明
OFF = 6        # 公休
YU = 7         # 有休

SHIFT_LABELS = {
    0: None, 1: "日", 2: "夜", 3: "明",
    4: "管夜", 5: "管明", 6: "休", 7: "有",
}
REQUEST_MAP = {
    "日": DAY, "夜": NIGHT, "明": AKE,
    "管夜": KAN_NIGHT, "管明": KAN_AKE, "休": OFF, "有": YU,
}
HARD_REQUESTS = {"休", "有"}
WORK_SHIFTS = {DAY, NIGHT, KAN_NIGHT}  # shifts that count as "working"


# ──────────────────────────────────────
# Night staffing requirement (weekly alternating pattern)
# ──────────────────────────────────────
def _build_night_req_table(
    year: int, month: int, num_days: int,
    night_pattern: list[int], start_with_three: bool = False,
) -> list[int]:
    """Return list[num_days] with the required night-shift count per day.

    The pattern alternates *weekly* (Sun-Sat weeks).
    Week boundaries are determined from the actual calendar.
    """
    first_dow = date(year, month + 1, 1).weekday()  # 0=Mon … 6=Sun
    # Convert to JS-style: 0=Sun, 1=Mon … 6=Sat
    first_dow_js = (first_dow + 1) % 7

    table: list[int] = []
    week_idx = 0
    days_until_sunday = (7 - first_dow_js) % 7

    # First partial week (if month doesn't start on Sunday)
    if days_until_sunday > 0 and days_until_sunday < 7:
        pat_idx = 0 if start_with_three else 1
        cnt = night_pattern[pat_idx % len(night_pattern)]
        for _ in range(min(days_until_sunday, num_days)):
            table.append(cnt)
        week_idx = 1

    # Full weeks
    while len(table) < num_days:
        if start_with_three:
            pat_idx = week_idx % len(night_pattern)
        else:
            pat_idx = (week_idx + 1) % len(night_pattern)
        cnt = night_pattern[pat_idx]
        for _ in range(7):
            if len(table) >= num_days:
                break
            table.append(cnt)
        week_idx += 1

    return table


def solve_schedule(request_data: dict) -> list[dict]:
    nurses = request_data["nurses"]
    num_days = request_data["daysInMonth"]
    year = request_data.get("year", 2026)
    month = request_data.get("month", 0)  # 0-indexed JS month
    config = request_data["config"]
    requests = request_data.get("requests", {})
    night_ng_pairs = request_data.get("nightNgPairs", [])
    prev_month = request_data.get("prevMonthConstraints", {})
    holidays = set(request_data.get("holidays", []))
    weekends = set(request_data.get("weekends", []))
    num_patterns = request_data.get("numPatterns", 3)

    weekday_day_staff = config["weekdayDayStaff"]
    weekend_day_staff = config["weekendDayStaff"]
    night_pattern = config.get("nightShiftPattern", [2, 2])
    max_night = config.get("maxNightShifts", 6)
    max_days_off = config.get("maxDaysOff", 10)
    max_consec = config.get("maxConsecutiveDays", 3)
    max_double_night = config.get("maxDoubleNightPairs", 2)

    active_nurses = [n for n in nurses if not n.get("excludeFromGeneration", False)]
    num_nurses = len(active_nurses)
    D = range(num_days)
    N = range(num_nurses)

    night_req_table = _build_night_req_table(
        year, month, num_days, night_pattern,
        start_with_three=config.get("startWithThree", False),
    )

    forbidden_solutions: list[dict] = []
    results: list[dict] = []
    pattern_labels = ["パターンA", "パターンB", "パターンC", "パターンD", "パターンE"]

    for pat_idx in range(num_patterns):
        model = cp_model.CpModel()

        # ── Variables ──────────────────────
        shifts: dict[tuple[int, int], cp_model.IntVar] = {}
        is_day: dict[tuple[int, int], cp_model.IntVar] = {}
        is_night: dict[tuple[int, int], cp_model.IntVar] = {}
        is_ake: dict[tuple[int, int], cp_model.IntVar] = {}
        is_kan_night: dict[tuple[int, int], cp_model.IntVar] = {}
        is_kan_ake: dict[tuple[int, int], cp_model.IntVar] = {}
        is_off: dict[tuple[int, int], cp_model.IntVar] = {}
        is_yu: dict[tuple[int, int], cp_model.IntVar] = {}
        is_working: dict[tuple[int, int], cp_model.IntVar] = {}

        for n in N:
            for d in D:
                # H1: domain 1..7 — no NULL, every cell must have a shift
                shifts[(n, d)] = model.new_int_var(1, 7, f"s_{n}_{d}")

                is_day[(n, d)] = model.new_bool_var(f"id_{n}_{d}")
                is_night[(n, d)] = model.new_bool_var(f"in_{n}_{d}")
                is_ake[(n, d)] = model.new_bool_var(f"ia_{n}_{d}")
                is_kan_night[(n, d)] = model.new_bool_var(f"ikn_{n}_{d}")
                is_kan_ake[(n, d)] = model.new_bool_var(f"ika_{n}_{d}")
                is_off[(n, d)] = model.new_bool_var(f"io_{n}_{d}")
                is_yu[(n, d)] = model.new_bool_var(f"iy_{n}_{d}")
                is_working[(n, d)] = model.new_bool_var(f"iw_{n}_{d}")

                # Link bools ↔ shift var
                model.add(shifts[(n, d)] == DAY).only_enforce_if(is_day[(n, d)])
                model.add(shifts[(n, d)] != DAY).only_enforce_if(is_day[(n, d)].negated())
                model.add(shifts[(n, d)] == NIGHT).only_enforce_if(is_night[(n, d)])
                model.add(shifts[(n, d)] != NIGHT).only_enforce_if(is_night[(n, d)].negated())
                model.add(shifts[(n, d)] == AKE).only_enforce_if(is_ake[(n, d)])
                model.add(shifts[(n, d)] != AKE).only_enforce_if(is_ake[(n, d)].negated())
                model.add(shifts[(n, d)] == KAN_NIGHT).only_enforce_if(is_kan_night[(n, d)])
                model.add(shifts[(n, d)] != KAN_NIGHT).only_enforce_if(is_kan_night[(n, d)].negated())
                model.add(shifts[(n, d)] == KAN_AKE).only_enforce_if(is_kan_ake[(n, d)])
                model.add(shifts[(n, d)] != KAN_AKE).only_enforce_if(is_kan_ake[(n, d)].negated())
                model.add(shifts[(n, d)] == OFF).only_enforce_if(is_off[(n, d)])
                model.add(shifts[(n, d)] != OFF).only_enforce_if(is_off[(n, d)].negated())
                model.add(shifts[(n, d)] == YU).only_enforce_if(is_yu[(n, d)])
                model.add(shifts[(n, d)] != YU).only_enforce_if(is_yu[(n, d)].negated())

                # is_working = DAY or NIGHT or KAN_NIGHT
                model.add_bool_or([is_day[(n, d)], is_night[(n, d)], is_kan_night[(n, d)]]).only_enforce_if(is_working[(n, d)])
                model.add_bool_and([is_day[(n, d)].negated(), is_night[(n, d)].negated(), is_kan_night[(n, d)].negated()]).only_enforce_if(is_working[(n, d)].negated())

        # ── Hard Constraints ──────────────

        for n in N:
            nurse = active_nurses[n]
            nurse_id = str(nurse["id"])
            nurse_reqs = requests.get(nurse_id, {})
            prev_nurse = prev_month.get(nurse_id, {})

            for d in D:
                # H2: 夜勤→翌日は明
                if d < num_days - 1:
                    model.add(shifts[(n, d + 1)] == AKE).only_enforce_if(is_night[(n, d)])
                # H3: 管夜→翌日は管明
                if d < num_days - 1:
                    model.add(shifts[(n, d + 1)] == KAN_AKE).only_enforce_if(is_kan_night[(n, d)])
                # H4: 明→翌日は休or有
                if d < num_days - 1:
                    ake_next_off = model.new_bool_var(f"ano_{n}_{d}")
                    ake_next_yu = model.new_bool_var(f"any_{n}_{d}")
                    model.add(shifts[(n, d + 1)] == OFF).only_enforce_if(ake_next_off)
                    model.add(shifts[(n, d + 1)] != OFF).only_enforce_if(ake_next_off.negated())
                    model.add(shifts[(n, d + 1)] == YU).only_enforce_if(ake_next_yu)
                    model.add(shifts[(n, d + 1)] != YU).only_enforce_if(ake_next_yu.negated())
                    model.add(ake_next_off + ake_next_yu >= 1).only_enforce_if(is_ake[(n, d)])
                # H5: 管明→翌日は休or有
                if d < num_days - 1:
                    kane_next_off = model.new_bool_var(f"kno_{n}_{d}")
                    kane_next_yu = model.new_bool_var(f"kny_{n}_{d}")
                    model.add(shifts[(n, d + 1)] == OFF).only_enforce_if(kane_next_off)
                    model.add(shifts[(n, d + 1)] != OFF).only_enforce_if(kane_next_off.negated())
                    model.add(shifts[(n, d + 1)] == YU).only_enforce_if(kane_next_yu)
                    model.add(shifts[(n, d + 1)] != YU).only_enforce_if(kane_next_yu.negated())
                    model.add(kane_next_off + kane_next_yu >= 1).only_enforce_if(is_kan_ake[(n, d)])

            # H13/H14: 月末の夜勤・管夜禁止
            # 最終日: 夜勤不可（翌日に明が必要）
            model.add(is_night[(n, num_days - 1)] == 0)
            model.add(is_kan_night[(n, num_days - 1)] == 0)
            # 最終日-1: 夜勤不可（明→休が月内に収まらない）
            if num_days >= 2:
                model.add(is_night[(n, num_days - 2)] == 0)
                model.add(is_kan_night[(n, num_days - 2)] == 0)

            # H6: 連続勤務日数制限
            window = max_consec + 1
            for d in D:
                if d + window <= num_days:
                    model.add(sum(is_working[(n, d + k)] for k in range(window)) <= max_consec)

            # H6 extended: 前月連続勤務引き継ぎ
            prev_consec = prev_nurse.get("_consecDays", 0)
            if prev_consec > 0:
                remaining = max_consec - prev_consec
                if remaining <= 0:
                    model.add(is_working[(n, 0)] == 0)
                else:
                    for end_d in range(1, min(remaining + 1, num_days)):
                        model.add(sum(is_working[(n, k)] for k in range(end_d + 1)) <= remaining)

            # H7: 3連夜勤禁止 (夜明夜明夜)
            for d in range(num_days - 4):
                model.add_bool_or([
                    is_night[(n, d)].negated(),
                    is_night[(n, d + 2)].negated(),
                    is_night[(n, d + 4)].negated(),
                ])

            # H9: 夜勤なし設定
            if nurse.get("noNightShift", False):
                for d in D:
                    model.add(is_night[(n, d)] == 0)
                    model.add(is_kan_night[(n, d)] == 0)

            # H10: 日勤なし設定
            if nurse.get("noDayShift", False):
                for d in D:
                    model.add(is_day[(n, d)] == 0)

            # H11: 希望の「休」「有」はハード制約
            for day_str, req_type in nurse_reqs.items():
                day_idx = int(day_str) - 1
                if 0 <= day_idx < num_days and req_type in HARD_REQUESTS:
                    target = REQUEST_MAP[req_type]
                    model.add(shifts[(n, day_idx)] == target)

            # H12: 前月制約の固定シフト
            for key, val in prev_nurse.items():
                if key.startswith("_"):
                    continue
                day_idx = int(key) - 1
                if 0 <= day_idx < num_days and val in REQUEST_MAP:
                    model.add(shifts[(n, day_idx)] == REQUEST_MAP[val])

            # H15: 夜勤回数上限（管夜はカウントしない）
            nurse_max_night = nurse.get("maxNightShifts", max_night)
            model.add(sum(is_night[(n, d)] for d in D) <= nurse_max_night)

        # H8: 夜勤NGペア
        for pair in night_ng_pairs:
            id_a, id_b = pair[0], pair[1]
            idx_a = next((i for i, nn in enumerate(active_nurses) if nn["id"] == id_a), None)
            idx_b = next((i for i, nn in enumerate(active_nurses) if nn["id"] == id_b), None)
            if idx_a is not None and idx_b is not None:
                for d in D:
                    model.add_bool_or([is_night[(idx_a, d)].negated(), is_night[(idx_b, d)].negated()])

        # ── Soft Constraints (Penalties) ──
        penalties: list[tuple] = []

        # S1: 各日の夜勤人数（管夜を除く）
        for d in D:
            required = night_req_table[d]
            total_night = sum(is_night[(n, d)] for n in N)
            over = model.new_int_var(0, num_nurses, f"no_{d}")
            under = model.new_int_var(0, num_nurses, f"nu_{d}")
            model.add(total_night - required == over - under)
            penalties.append((over, 1000))
            penalties.append((under, 1000))

        # S2: 各日の日勤人数
        for d in D:
            required = weekend_day_staff if d in weekends else weekday_day_staff
            total_day = sum(is_day[(n, d)] for n in N)
            over = model.new_int_var(0, num_nurses, f"do_{d}")
            under = model.new_int_var(0, num_nurses, f"du_{d}")
            model.add(total_day - required == over - under)
            penalties.append((over, 500))
            penalties.append((under, 500))

        # S3: 夜勤回数の均等化（管夜を除く）
        night_counts = []
        for n in N:
            nc = model.new_int_var(0, num_days, f"nc_{n}")
            model.add(nc == sum(is_night[(n, d)] for d in D))
            night_counts.append(nc)
        if len(night_counts) > 1:
            max_nc = model.new_int_var(0, num_days, "max_nc")
            min_nc = model.new_int_var(0, num_days, "min_nc")
            model.add_max_equality(max_nc, night_counts)
            model.add_min_equality(min_nc, night_counts)
            diff = model.new_int_var(0, num_days, "nc_diff")
            model.add(diff == max_nc - min_nc)
            penalties.append((diff, 300))

        # S4: 夜勤の月内分散
        seg_len = num_days // 3
        for n in N:
            segs = []
            for s in range(3):
                start = s * seg_len
                end = (s + 1) * seg_len if s < 2 else num_days
                sc = model.new_int_var(0, num_days, f"seg_{n}_{s}")
                model.add(sc == sum(is_night[(n, d)] for d in range(start, end)))
                segs.append(sc)
            for i in range(3):
                for j in range(i + 1, 3):
                    d_var = model.new_int_var(0, num_days, f"sd_{n}_{i}_{j}")
                    model.add_abs_equality(d_var, segs[i] - segs[j])
                    penalties.append((d_var, 50))

        # S5: 2連夜勤ペア上限
        for n in N:
            dbl_bools = []
            for d in range(num_days - 2):
                b = model.new_bool_var(f"db_{n}_{d}")
                model.add_bool_and([is_night[(n, d)], is_night[(n, d + 2)]]).only_enforce_if(b)
                model.add_bool_or([is_night[(n, d)].negated(), is_night[(n, d + 2)].negated()]).only_enforce_if(b.negated())
                dbl_bools.append(b)
            dbl_count = model.new_int_var(0, num_days, f"dbc_{n}")
            model.add(dbl_count == sum(dbl_bools))
            excess = model.new_int_var(0, num_days, f"dbe_{n}")
            model.add(excess >= dbl_count - max_double_night)
            model.add(excess >= 0)
            penalties.append((excess, 500))

        # S6: 休日数（休+有のみ。明・管明は含めない）
        for n in N:
            off_cnt = model.new_int_var(0, num_days, f"oc_{n}")
            model.add(off_cnt == sum(is_off[(n, d)] + is_yu[(n, d)] for d in D))
            off_over = model.new_int_var(0, num_days, f"oo_{n}")
            off_under = model.new_int_var(0, num_days, f"ou_{n}")
            model.add(off_cnt - max_days_off == off_over - off_under)
            penalties.append((off_over, 200))
            penalties.append((off_under, 200))

        # S7: 希望の「日」「夜」等はソフト制約
        soft_bonuses: list = []
        for n in N:
            nurse_id = str(active_nurses[n]["id"])
            nurse_reqs = requests.get(nurse_id, {})
            for day_str, req_type in nurse_reqs.items():
                day_idx = int(day_str) - 1
                if 0 <= day_idx < num_days and req_type not in HARD_REQUESTS and req_type in REQUEST_MAP:
                    matched = model.new_bool_var(f"sr_{n}_{day_idx}")
                    model.add(shifts[(n, day_idx)] == REQUEST_MAP[req_type]).only_enforce_if(matched)
                    model.add(shifts[(n, day_idx)] != REQUEST_MAP[req_type]).only_enforce_if(matched.negated())
                    soft_bonuses.append(matched)

        # ── Forbid previous solutions ──
        for forbidden in forbidden_solutions:
            diffs = []
            for n in N:
                nid = str(active_nurses[n]["id"])
                if nid in forbidden:
                    for d in D:
                        b = model.new_bool_var(f"fb_{pat_idx}_{n}_{d}")
                        model.add(shifts[(n, d)] != forbidden[nid][d]).only_enforce_if(b)
                        model.add(shifts[(n, d)] == forbidden[nid][d]).only_enforce_if(b.negated())
                        diffs.append(b)
            if diffs:
                model.add(sum(diffs) >= max(1, num_nurses))  # require meaningful difference

        # ── Objective ──
        total_penalty = sum(var * w for var, w in penalties)
        bonus = sum(b * 200 for b in soft_bonuses) if soft_bonuses else 0
        model.minimize(total_penalty - bonus)

        # ── Solve ──
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        solver.parameters.num_workers = 4

        status = solver.solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution_data: dict[str, list] = {}
            solution_raw: dict[str, list[int]] = {}
            for n in N:
                nid = str(active_nurses[n]["id"])
                raw = []
                for d in D:
                    v = solver.value(shifts[(n, d)])
                    if v < 1 or v > 7:
                        v = OFF  # フォールバック: 不正値は休にする
                    raw.append(v)
                solution_data[nid] = [SHIFT_LABELS.get(v, "休") for v in raw]
                solution_raw[nid] = raw

            forbidden_solutions.append(solution_raw)

            # ── Metrics ──
            night_vals = []
            total_req = 0
            matched_req = 0
            total_night_shortage = 0
            total_day_shortage = 0
            consec_violations = 0
            null_cells = 0

            for n in N:
                nid = str(active_nurses[n]["id"])
                nc = sum(1 for d in D if solver.value(shifts[(n, d)]) == NIGHT)
                night_vals.append(nc)

                # Request matching
                nr = requests.get(nid, {})
                for ds, rt in nr.items():
                    di = int(ds) - 1
                    if 0 <= di < num_days and rt in REQUEST_MAP:
                        total_req += 1
                        if solver.value(shifts[(n, di)]) == REQUEST_MAP[rt]:
                            matched_req += 1

                # Consecutive violations
                consec = 0
                for d in D:
                    v = solver.value(shifts[(n, d)])
                    if v in WORK_SHIFTS:
                        consec += 1
                        if consec > max_consec:
                            consec_violations += 1
                    else:
                        consec = 0

                # Null cell check
                for d in D:
                    label = SHIFT_LABELS.get(solver.value(shifts[(n, d)]))
                    if label is None or label == "":
                        null_cells += 1

            for d in D:
                actual_night = sum(1 for n in N if solver.value(shifts[(n, d)]) == NIGHT)
                if actual_night < night_req_table[d]:
                    total_night_shortage += night_req_table[d] - actual_night
                req_day = weekend_day_staff if d in weekends else weekday_day_staff
                actual_day = sum(1 for n in N if solver.value(shifts[(n, d)]) == DAY)
                if actual_day < req_day:
                    total_day_shortage += req_day - actual_day

            night_balance = (max(night_vals) - min(night_vals)) if night_vals else 0
            req_match = (matched_req / total_req * 100) if total_req > 0 else 100.0

            # Compute off days (休+有 only)
            total_off = sum(
                sum(1 for d in D if solver.value(shifts[(n, d)]) in (OFF, YU))
                for n in N
            )
            avg_off = total_off / num_nurses if num_nurses > 0 else 0

            score = 10000 - solver.objective_value

            results.append({
                "label": pattern_labels[pat_idx] if pat_idx < len(pattern_labels) else f"パターン{pat_idx + 1}",
                "data": solution_data,
                "score": int(score),
                "metrics": {
                    "nightBalance": round(night_balance, 1),
                    "dayShortage": total_day_shortage,
                    "nightShortage": total_night_shortage,
                    "consecViolations": consec_violations,
                    "requestMatch": round(req_match, 1),
                    "avgDaysOff": round(avg_off, 1),
                    "nullCells": null_cells,
                },
            })
        else:
            status_name = solver.status_name(status)
            if status_name == "INFEASIBLE":
                err_msg = "制約が厳しすぎて解が存在しません。希望や夜勤回数上限を見直してください"
            elif status_name == "MODEL_INVALID":
                err_msg = "モデルが不正です。入力データを確認してください"
            elif status_name == "UNKNOWN":
                err_msg = "時間内に解が見つかりませんでした。制約を緩めるか再実行してください"
            else:
                err_msg = f"Solver status: {status_name}"
            results.append({
                "label": pattern_labels[pat_idx] if pat_idx < len(pattern_labels) else f"パターン{pat_idx + 1}",
                "data": {},
                "score": 0,
                "metrics": {"error": err_msg, "solverStatus": status_name},
            })

    return results
