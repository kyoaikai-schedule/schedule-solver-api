"""OR-Tools CP-SAT nurse schedule solver — v3 (3-shift core + post-processing)."""

from __future__ import annotations
from datetime import date
from ortools.sat.python import cp_model

# ──────────────────────────────────────
# Solver-internal shift values (3 only)
# ──────────────────────────────────────
DAY = 1     # 日勤
NIGHT = 2   # 夜勤
OFF = 3     # 非勤務 (post-procで休/明/管明/有に変換)

SOLVER_VAL_TO_LABEL = {DAY: "日", NIGHT: "夜", OFF: "休"}

# 入力ラベル → solver値
LABEL_TO_SOLVER = {
    "日": DAY,
    "夜": NIGHT,
    "管夜": NIGHT,   # solverではNIGHT扱い、ただし夜勤人数カウントから除外
    "明": OFF,
    "管明": OFF,
    "休": OFF,
    "有": OFF,
}

WORK_LABELS = {"日", "夜", "管夜"}


# ──────────────────────────────────────
# Helpers
# ──────────────────────────────────────
def _build_night_req_table(
    year: int, month: int, num_days: int,
    night_pattern: list[int], start_with_three: bool = False,
) -> list[int]:
    first_dow = date(year, month + 1, 1).weekday()
    first_dow_js = (first_dow + 1) % 7
    table: list[int] = []
    week_idx = 0
    days_until_sunday = (7 - first_dow_js) % 7
    if 0 < days_until_sunday < 7:
        pat_idx = 0 if start_with_three else 1
        cnt = night_pattern[pat_idx % len(night_pattern)]
        for _ in range(min(days_until_sunday, num_days)):
            table.append(cnt)
        week_idx = 1
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


def _build_forced(active_nurses, requests, prev_month, num_days):
    """Build forced_shift (solver value) and forced_label (output label) maps.

    Auto-fills chains:
      管夜@d → 管明@d+1, 休@d+2
      管明@d → 休@d+1
      明 @d → 休@d+1 (forced 明は通常2連にしない安全寄り)
      夜 @d → solver制約で自動的にOFF@d+1, OFF@d+2 候補（post-procで明/休に変換）
    """
    forced_shift: dict[tuple[int, int], int] = {}
    forced_label: dict[tuple[int, int], str] = {}

    def add(n, d, val, label, *, override=False):
        if not (0 <= d < num_days):
            return
        if (n, d) in forced_shift and not override:
            return
        forced_shift[(n, d)] = val
        forced_label[(n, d)] = label

    def apply(n, d, label, *, override=False):
        if label == "管夜":
            add(n, d, NIGHT, "管夜", override=override)
            add(n, d + 1, OFF, "管明")
            add(n, d + 2, OFF, "休")
        elif label == "管明":
            add(n, d, OFF, "管明", override=override)
            add(n, d + 1, OFF, "休")
        elif label == "明":
            add(n, d, OFF, "明", override=override)
            add(n, d + 1, OFF, "休")
        elif label == "夜":
            add(n, d, NIGHT, "夜", override=override)
        elif label in ("日", "休", "有"):
            add(n, d, LABEL_TO_SOLVER[label], label, override=override)

    for n_idx, nurse in enumerate(active_nurses):
        nid = str(nurse["id"])

        # 1. 希望（最優先）
        for day_str, label in requests.get(nid, {}).items():
            try:
                d = int(day_str) - 1
            except (TypeError, ValueError):
                continue
            apply(n_idx, d, label, override=True)

        # 2. 前月制約（希望と被ったらスキップ）
        prev = prev_month.get(nid, {})
        for key, label in prev.items():
            if key.startswith("_"):
                continue
            try:
                d = int(key) - 1
            except (TypeError, ValueError):
                continue
            apply(n_idx, d, label)

    return forced_shift, forced_label


# ──────────────────────────────────────
# Solver core
# ──────────────────────────────────────
def _solve_one_pattern(params, forbidden_solutions, relax_level=0):
    active_nurses = params["active_nurses"]
    num_days = params["num_days"]
    night_req_table = params["night_req_table"]
    weekday_day_staff = params["weekday_day_staff"]
    weekend_day_staff = params["weekend_day_staff"]
    weekends = params["weekends"]
    max_consec = params["max_consec"]
    max_night = params["max_night"]
    max_days_off = params["max_days_off"]
    max_double_night = params["max_double_night"]
    forced_shift = params["forced_shift"]
    forced_label = params["forced_label"]
    prev_month = params["prev_month"]
    night_ng_pairs = params["night_ng_pairs"]

    num_nurses = len(active_nurses)
    N = range(num_nurses)
    D = range(num_days)

    model = cp_model.CpModel()

    shifts = {}
    is_day = {}
    is_night = {}
    is_off = {}
    is_working = {}

    for n in N:
        for d in D:
            shifts[(n, d)] = model.new_int_var(1, 3, f"s_{n}_{d}")
            is_day[(n, d)] = model.new_bool_var(f"id_{n}_{d}")
            is_night[(n, d)] = model.new_bool_var(f"in_{n}_{d}")
            is_off[(n, d)] = model.new_bool_var(f"io_{n}_{d}")
            is_working[(n, d)] = model.new_bool_var(f"iw_{n}_{d}")

            model.add(shifts[(n, d)] == DAY).only_enforce_if(is_day[(n, d)])
            model.add(shifts[(n, d)] != DAY).only_enforce_if(is_day[(n, d)].negated())
            model.add(shifts[(n, d)] == NIGHT).only_enforce_if(is_night[(n, d)])
            model.add(shifts[(n, d)] != NIGHT).only_enforce_if(is_night[(n, d)].negated())
            model.add(shifts[(n, d)] == OFF).only_enforce_if(is_off[(n, d)])
            model.add(shifts[(n, d)] != OFF).only_enforce_if(is_off[(n, d)].negated())

            model.add(is_day[(n, d)] + is_night[(n, d)] + is_off[(n, d)] == 1)
            model.add(is_working[(n, d)] == 1 - is_off[(n, d)])

    # 固定セル
    for (n, d), val in forced_shift.items():
        model.add(shifts[(n, d)] == val)

    # ── ハード制約: 夜→翌日OFF / 夜→翌々日 NOT DAY ──
    for n in N:
        for d in D:
            if d + 1 < num_days:
                model.add(is_off[(n, d + 1)] == 1).only_enforce_if(is_night[(n, d)])
            if d + 2 < num_days:
                model.add(is_day[(n, d + 2)] == 0).only_enforce_if(is_night[(n, d)])

    # ── ハード制約: forced 休/有 の前日に NIGHT を置かない ──
    # (置くと post-proc で 夜→休 となり 夜→明 invariant が崩れる)
    for (n, d), lbl in forced_label.items():
        if lbl in ("休", "有") and d > 0 and (n, d - 1) not in forced_shift:
            model.add(is_night[(n, d - 1)] == 0)

    # ── 3連夜勤禁止 (NIGHT @ d, d+2, d+4) ──
    for n in N:
        for d in range(num_days - 4):
            model.add_bool_or([
                is_night[(n, d)].negated(),
                is_night[(n, d + 2)].negated(),
                is_night[(n, d + 4)].negated(),
            ])

    # ── 月末夜勤禁止 (forced夜は除く) ──
    for n in N:
        for d in (num_days - 1, num_days - 2):
            if d < 0 or d >= num_days:
                continue
            if forced_shift.get((n, d)) == NIGHT:
                continue
            model.add(is_night[(n, d)] == 0)

    # ── 連続勤務制限 ──
    for n in N:
        nurse = active_nurses[n]
        nid = str(nurse["id"])
        prev_consec = prev_month.get(nid, {}).get("_consecDays", 0)

        window = max_consec + 1
        for d in D:
            if d + window <= num_days:
                model.add(sum(is_working[(n, d + k)] for k in range(window)) <= max_consec)

        if prev_consec > 0:
            remaining = max_consec - prev_consec
            if remaining <= 0:
                if 0 < num_days:
                    model.add(is_working[(n, 0)] == 0)
            else:
                limit = min(remaining + 1, num_days)
                if limit > 0:
                    model.add(sum(is_working[(n, k)] for k in range(limit)) <= remaining)

    # ── 夜勤NGペア ──
    for pair in night_ng_pairs:
        if len(pair) < 2:
            continue
        id_a, id_b = pair[0], pair[1]
        idx_a = next((i for i, nn in enumerate(active_nurses) if nn["id"] == id_a), None)
        idx_b = next((i for i, nn in enumerate(active_nurses) if nn["id"] == id_b), None)
        if idx_a is not None and idx_b is not None:
            for d in D:
                model.add_bool_or([is_night[(idx_a, d)].negated(), is_night[(idx_b, d)].negated()])

    # ── 個人設定 ──
    for n in N:
        nurse = active_nurses[n]
        if nurse.get("noNightShift", False):
            for d in D:
                if forced_shift.get((n, d)) != NIGHT:
                    model.add(is_night[(n, d)] == 0)
        if nurse.get("noDayShift", False):
            for d in D:
                if forced_shift.get((n, d)) != DAY:
                    model.add(is_day[(n, d)] == 0)

        nurse_max_night = nurse.get("maxNightShifts", max_night)
        real_night_terms = [is_night[(n, d)] for d in D if forced_label.get((n, d)) != "管夜"]
        if real_night_terms:
            model.add(sum(real_night_terms) <= nurse_max_night)

    # ── 夜勤人数（ハード／緩和あり）──
    night_dev_penalties = []
    for d in D:
        required = night_req_table[d]
        terms = [is_night[(n, d)] for n in N if forced_label.get((n, d)) != "管夜"]
        if not terms:
            continue
        total = sum(terms)
        if relax_level == 0:
            model.add(total == required)
        else:
            model.add(total >= max(0, required - 1))
            model.add(total <= required + 1)
            # 緩和時も要件への追従を強くペナルティ
            diff = model.new_int_var(-num_nurses, num_nurses, f"nd_{d}")
            model.add(diff == total - required)
            abs_diff = model.new_int_var(0, num_nurses, f"and_{d}")
            model.add_abs_equality(abs_diff, diff)
            night_dev_penalties.append((abs_diff, 5000))

    # ── 日勤人数（ハード／緩和あり）──
    day_short_penalties = []
    for d in D:
        required = weekend_day_staff if d in weekends else weekday_day_staff
        total_day = sum(is_day[(n, d)] for n in N)
        if relax_level == 0:
            model.add(total_day >= required)
        else:
            model.add(total_day >= max(0, required - 1))
            short = model.new_int_var(0, num_nurses, f"ds_{d}")
            model.add(short >= required - total_day)
            day_short_penalties.append((short, 5000))

    # ──────────────────────────────────
    # ソフト制約（ペナルティ）
    # ──────────────────────────────────
    penalties: list[tuple] = []
    penalties.extend(night_dev_penalties)
    penalties.extend(day_short_penalties)

    # 夜勤回数の均等化（管夜除く）
    night_counts = []
    for n in N:
        nc = model.new_int_var(0, num_days, f"nc_{n}")
        terms = [is_night[(n, d)] for d in D if forced_label.get((n, d)) != "管夜"]
        if terms:
            model.add(nc == sum(terms))
        else:
            model.add(nc == 0)
        night_counts.append(nc)
    if len(night_counts) > 1:
        max_nc = model.new_int_var(0, num_days, "max_nc")
        min_nc = model.new_int_var(0, num_days, "min_nc")
        model.add_max_equality(max_nc, night_counts)
        model.add_min_equality(min_nc, night_counts)
        diff = model.new_int_var(0, num_days, "nc_diff")
        model.add(diff == max_nc - min_nc)
        penalties.append((diff, 300))

    # 月内分散
    if num_days >= 6:
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

    # 2連夜勤ペア上限
    for n in N:
        dbl_bools = []
        for d in range(num_days - 2):
            b = model.new_bool_var(f"db_{n}_{d}")
            model.add_bool_and([is_night[(n, d)], is_night[(n, d + 2)]]).only_enforce_if(b)
            model.add_bool_or([is_night[(n, d)].negated(), is_night[(n, d + 2)].negated()]).only_enforce_if(b.negated())
            dbl_bools.append(b)
        if dbl_bools:
            dbl_count = model.new_int_var(0, num_days, f"dbc_{n}")
            model.add(dbl_count == sum(dbl_bools))
            excess = model.new_int_var(0, num_days, f"dbe_{n}")
            model.add(excess >= dbl_count - max_double_night)
            model.add(excess >= 0)
            penalties.append((excess, 500))

    # 休日数(休+有 のみ。明・管明は除外)
    # 明候補 = is_off@d AND (is_night@d-1 OR forced_label[d] in {明,管明})
    # まず post-proc で 明/管明 になる cell の数を計算する。
    for n in N:
        ake_kanake_terms = []
        for d in D:
            forced = forced_label.get((n, d))
            if forced in ("明", "管明"):
                ake_kanake_terms.append(1)
            elif forced is not None:
                # 休/有/日/夜/管夜 — 明/管明にはならない
                continue
            else:
                if d > 0:
                    # 自動: 前日が NIGHT なら明 or 管明
                    ake_kanake_terms.append(is_night[(n, d - 1)])
        ak_total = model.new_int_var(0, num_days, f"ak_{n}")
        if ake_kanake_terms:
            model.add(ak_total == sum(ake_kanake_terms))
        else:
            model.add(ak_total == 0)

        total_off = sum(is_off[(n, d)] for d in D)
        real_off = model.new_int_var(0, num_days, f"ro_{n}")
        model.add(real_off == total_off - ak_total)

        off_over = model.new_int_var(0, num_days, f"oo_{n}")
        off_under = model.new_int_var(0, num_days, f"ou_{n}")
        model.add(real_off - max_days_off == off_over - off_under)
        penalties.append((off_over, 200))
        penalties.append((off_under, 200))

    # ── 既出解の禁止 ──
    for fb_idx, forbidden in enumerate(forbidden_solutions):
        diffs = []
        for n in N:
            nid = str(active_nurses[n]["id"])
            if nid not in forbidden:
                continue
            for d in D:
                if (n, d) in forced_shift:
                    continue
                b = model.new_bool_var(f"fb_{fb_idx}_{n}_{d}")
                model.add(shifts[(n, d)] != forbidden[nid][d]).only_enforce_if(b)
                model.add(shifts[(n, d)] == forbidden[nid][d]).only_enforce_if(b.negated())
                diffs.append(b)
        if diffs:
            model.add(sum(diffs) >= max(1, num_nurses // 2))

    # ── Objective ──
    if penalties:
        total_penalty = sum(var * w for var, w in penalties)
        model.minimize(total_penalty)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 30
    solver.parameters.num_workers = 4

    status = solver.solve(model)
    status_name = solver.status_name(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raw = {}
        for n in N:
            nid = str(active_nurses[n]["id"])
            raw[nid] = [solver.value(shifts[(n, d)]) for d in D]
        return {
            "raw": raw,
            "objective": int(solver.objective_value) if penalties else 0,
            "status": status_name,
        }
    return {"raw": None, "objective": None, "status": status_name}


# ──────────────────────────────────────
# Post-processing & validation
# ──────────────────────────────────────
def _post_process(solution_raw, active_nurses, forced_label, num_days):
    """Convert solver values (DAY/NIGHT/OFF) to display labels."""
    output = {}
    for n_idx, nurse in enumerate(active_nurses):
        nid = str(nurse["id"])
        labels: list[str] = [None] * num_days  # type: ignore
        for d in range(num_days):
            forced = forced_label.get((n_idx, d))
            if forced is not None:
                labels[d] = forced
                continue
            v = solution_raw[nid][d]
            if v == DAY:
                labels[d] = "日"
            elif v == NIGHT:
                labels[d] = "夜"
            else:  # OFF
                if d > 0:
                    prev = labels[d - 1]
                    if prev == "夜":
                        labels[d] = "明"
                    elif prev == "管夜":
                        labels[d] = "管明"
                    else:
                        labels[d] = "休"
                else:
                    labels[d] = "休"
        output[nid] = labels
    return output


def _validate(data, params):
    errors = []
    num_days = params["num_days"]
    night_req_table = params["night_req_table"]
    weekday_day_staff = params["weekday_day_staff"]
    weekend_day_staff = params["weekend_day_staff"]
    weekends = params["weekends"]
    max_consec = params["max_consec"]
    max_night = params["max_night"]
    active_nurses = params["active_nurses"]

    for nid, sh in data.items():
        for d, s in enumerate(sh):
            if s is None or s == "":
                errors.append(f"Nurse {nid} Day {d+1}: 空白セル")
            if s == "夜" and d + 1 < len(sh) and sh[d + 1] != "明":
                errors.append(f"Nurse {nid} Day {d+1}: 夜の翌日が明でない({sh[d+1]})")
            if s == "管夜" and d + 1 < len(sh) and sh[d + 1] != "管明":
                errors.append(f"Nurse {nid} Day {d+1}: 管夜の翌日が管明でない({sh[d+1]})")
            if s == "明" and d + 1 < len(sh) and sh[d + 1] not in ("休", "夜", "有"):
                errors.append(f"Nurse {nid} Day {d+1}: 明の翌日が休/夜/有以外({sh[d+1]})")
            if s == "管明" and d + 1 < len(sh) and sh[d + 1] not in ("休", "有"):
                errors.append(f"Nurse {nid} Day {d+1}: 管明の翌日が休/有以外({sh[d+1]})")

    for d in range(num_days):
        actual = sum(1 for sh in data.values() if sh[d] == "夜")
        if actual != night_req_table[d]:
            errors.append(f"Day {d+1}: 夜勤人数 {actual} ≠ 要件 {night_req_table[d]}")

    for d in range(num_days):
        actual = sum(1 for sh in data.values() if sh[d] == "日")
        req = weekend_day_staff if d in weekends else weekday_day_staff
        if actual < req:
            errors.append(f"Day {d+1}: 日勤人数 {actual} < 要件 {req}")

    for nid, sh in data.items():
        consec = 0
        for d, s in enumerate(sh):
            if s in WORK_LABELS:
                consec += 1
                if consec > max_consec:
                    errors.append(f"Nurse {nid} Day {d+1}: 連続勤務 {consec}日 (上限{max_consec})")
            else:
                consec = 0

    for nurse in active_nurses:
        nid = str(nurse["id"])
        if nid not in data:
            continue
        nurse_max = nurse.get("maxNightShifts", max_night)
        cnt = sum(1 for s in data[nid] if s == "夜")
        if cnt > nurse_max:
            errors.append(f"Nurse {nid}: 夜勤 {cnt}回 (上限{nurse_max})")

    return errors


# ──────────────────────────────────────
# Main entry
# ──────────────────────────────────────
def solve_schedule(request_data: dict) -> list[dict]:
    nurses = request_data["nurses"]
    num_days = request_data["daysInMonth"]
    year = request_data.get("year", 2026)
    month = request_data.get("month", 0)
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
    night_req_table = _build_night_req_table(
        year, month, num_days, night_pattern,
        start_with_three=config.get("startWithThree", False),
    )
    # 月末2日は夜勤禁止 → 必要人数も0で整合させる
    for d in (num_days - 1, num_days - 2):
        if 0 <= d < num_days:
            night_req_table[d] = 0

    forced_shift, forced_label = _build_forced(active_nurses, requests, prev_month, num_days)

    weekends_combined = weekends | holidays

    params = {
        "active_nurses": active_nurses,
        "num_days": num_days,
        "night_req_table": night_req_table,
        "weekday_day_staff": weekday_day_staff,
        "weekend_day_staff": weekend_day_staff,
        "weekends": weekends_combined,
        "max_consec": max_consec,
        "max_night": max_night,
        "max_days_off": max_days_off,
        "max_double_night": max_double_night,
        "forced_shift": forced_shift,
        "forced_label": forced_label,
        "prev_month": prev_month,
        "night_ng_pairs": night_ng_pairs,
    }

    forbidden_solutions: list[dict] = []
    results: list[dict] = []
    pattern_labels = ["パターンA", "パターンB", "パターンC", "パターンD", "パターンE"]

    for pat_idx in range(num_patterns):
        label = pattern_labels[pat_idx] if pat_idx < len(pattern_labels) else f"パターン{pat_idx + 1}"

        chosen = None
        chosen_errors: list[str] = []
        last_status = None

        # 試行: relax 0 → 1。各レベルで最大3回（解後validation失敗時のリトライ）
        for relax_level in (0, 1):
            for attempt in range(3):
                res = _solve_one_pattern(params, forbidden_solutions, relax_level=relax_level)
                last_status = res["status"]
                if res["raw"] is None:
                    break  # この緩和では解なし、次の緩和へ
                final_data = _post_process(res["raw"], active_nurses, forced_label, num_days)
                errors = _validate(final_data, params)
                if not errors:
                    chosen = {
                        "raw": res["raw"],
                        "data": final_data,
                        "objective": res["objective"],
                        "relax_level": relax_level,
                    }
                    break
                # 同じ条件でリトライしても結果は同じなので即break
                chosen_errors = errors
                break
            if chosen is not None:
                break

        if chosen is None:
            results.append({
                "label": label,
                "data": {},
                "score": 0,
                "metrics": {
                    "error": f"解が見つかりませんでした (status={last_status})",
                    "solverStatus": last_status,
                    "validationErrors": chosen_errors[:10],
                },
            })
            continue

        forbidden_solutions.append(chosen["raw"])
        data = chosen["data"]

        # ── Metrics ──
        night_vals = [sum(1 for s in data[str(n["id"])] if s == "夜") for n in active_nurses]
        total_night_shortage = 0
        total_day_shortage = 0
        for d in range(num_days):
            actual_night = sum(1 for sh in data.values() if sh[d] == "夜")
            req_night = night_req_table[d]
            if actual_night < req_night:
                total_night_shortage += req_night - actual_night
            req_day = weekend_day_staff if d in weekends_combined else weekday_day_staff
            actual_day = sum(1 for sh in data.values() if sh[d] == "日")
            if actual_day < req_day:
                total_day_shortage += req_day - actual_day

        consec_violations = 0
        null_cells = 0
        for nid, sh in data.items():
            consec = 0
            for d, s in enumerate(sh):
                if s in WORK_LABELS:
                    consec += 1
                    if consec > max_consec:
                        consec_violations += 1
                else:
                    consec = 0
                if s is None or s == "":
                    null_cells += 1

        total_req = 0
        matched_req = 0
        for nid_str, reqs in requests.items():
            for day_str, req_type in reqs.items():
                try:
                    d = int(day_str) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= d < num_days and nid_str in data:
                    total_req += 1
                    if data[nid_str][d] == req_type:
                        matched_req += 1
        req_match = (matched_req / total_req * 100) if total_req > 0 else 100.0

        total_off = sum(sum(1 for s in sh if s in ("休", "有")) for sh in data.values())
        avg_off = total_off / len(active_nurses) if active_nurses else 0
        night_balance = (max(night_vals) - min(night_vals)) if night_vals else 0

        score = max(0, 10000 - (chosen["objective"] or 0))

        results.append({
            "label": label,
            "data": data,
            "score": int(score),
            "metrics": {
                "nightBalance": round(night_balance, 1),
                "dayShortage": total_day_shortage,
                "nightShortage": total_night_shortage,
                "consecViolations": consec_violations,
                "requestMatch": round(req_match, 1),
                "avgDaysOff": round(avg_off, 1),
                "nullCells": null_cells,
                "relaxLevel": chosen["relax_level"],
            },
        })

    return results
