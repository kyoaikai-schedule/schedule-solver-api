"""OR-Tools CP-SAT nurse schedule solver — v3 (3-shift core + post-processing)."""

from __future__ import annotations
import sys
import time
from datetime import date
from ortools.sat.python import cp_model


def _log(msg: str) -> None:
    """Cloud Run captures stdout/stderr → そのまま logs に流れる。"""
    print(f"[solver] {msg}", file=sys.stderr, flush=True)

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
# Infeasibility 解析: 制約グループに assumption literal を付け、
# UNSAT core (SufficientAssumptionsForInfeasibility) を抽出
# ──────────────────────────────────────
def _diagnose_infeasible(params):
    """各制約グループに assumption を付与して INFEASIBLE の主因を特定する。

    通常 solve よりずっと小さなモデル: 緩和 level 2 と同じ条件で
    assumption-conditional に組み立てる。返り値は活発な
    制約グループ名のリスト (UNSAT に効いた制約)。"""
    active_nurses = params["active_nurses"]
    num_days = params["num_days"]
    night_req_table = params["night_req_table"]
    weekday_day_staff = params["weekday_day_staff"]
    weekend_day_staff = params["weekend_day_staff"]
    weekends = params["weekends"]
    max_consec = params["max_consec"]
    max_night = params["max_night"]
    forced_shift = params["forced_shift"]
    forced_label = params["forced_label"]
    prev_month = params["prev_month"]
    night_ng_pairs = params["night_ng_pairs"]

    num_nurses = len(active_nurses)
    if num_nurses == 0 or num_days == 0:
        return {"sufficientAssumptions": [], "note": "no nurses or zero days"}

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

    # 各制約グループ用 assumption literal
    # (Trueに固定したいので、各グループのenforceに使う)
    assumptions: dict[str, cp_model.IntVar] = {}

    def make_assumption(name: str):
        a = model.new_bool_var(f"a_{name}")
        assumptions[name] = a
        return a

    # ── A. NIGHT→OFF / NIGHT→OFF (翌々日も) ──
    a_chain = make_assumption("nightChain")
    for n in N:
        for d in D:
            if d + 1 < num_days:
                # is_night→is_off@d+1, conditional on a_chain
                # ⇔ NOT a_chain OR NOT is_night OR is_off
                model.add_bool_or([a_chain.negated(), is_night[(n, d)].negated(), is_off[(n, d + 1)]])
            if d + 2 < num_days:
                model.add_bool_or([a_chain.negated(), is_night[(n, d)].negated(), is_off[(n, d + 2)]])

    # ── B. 3連夜勤禁止 ──
    a_no_triple = make_assumption("noTripleNight")
    for n in N:
        for d in range(num_days - 4):
            model.add_bool_or([
                a_no_triple.negated(),
                is_night[(n, d)].negated(),
                is_night[(n, d + 2)].negated(),
                is_night[(n, d + 4)].negated(),
            ])

    # ── C. 連続勤務 ──
    a_consec = make_assumption("maxConsec")
    window = max_consec + 1
    for n in N:
        for d in D:
            if d + window <= num_days:
                # sum(is_working[d..d+window-1]) ≤ max_consec, conditional on a_consec
                # ⇔ if a_consec then sum ≤ max_consec
                model.add(sum(is_working[(n, d + k)] for k in range(window)) <= max_consec).only_enforce_if(a_consec)

    # ── D. 前月引継ぎ連続勤務 ──
    a_prev_consec = make_assumption("prevConsec")
    for n in N:
        nurse = active_nurses[n]
        nid = str(nurse["id"])
        prev_consec = prev_month.get(nid, {}).get("_consecDays", 0) if isinstance(prev_month, dict) else 0
        if prev_consec > 0:
            remaining = max_consec - prev_consec
            if remaining <= 0:
                if 0 < num_days:
                    model.add(is_working[(n, 0)] == 0).only_enforce_if(a_prev_consec)
            else:
                limit = min(remaining + 1, num_days)
                if limit > 0:
                    model.add(sum(is_working[(n, k)] for k in range(limit)) <= remaining).only_enforce_if(a_prev_consec)

    # ── E. 個別 noNightShift / noDayShift ──
    a_no_night_pref = make_assumption("noNightPref")
    a_no_day_pref = make_assumption("noDayPref")
    for n in N:
        nurse = active_nurses[n]
        if nurse.get("noNightShift", False):
            for d in D:
                if forced_shift.get((n, d)) != NIGHT:
                    model.add(is_night[(n, d)] == 0).only_enforce_if(a_no_night_pref)
        if nurse.get("noDayShift", False):
            for d in D:
                if forced_shift.get((n, d)) != DAY:
                    model.add(is_day[(n, d)] == 0).only_enforce_if(a_no_day_pref)

    # ── F. 個別 maxNightShifts ──
    a_max_night = make_assumption("maxNightShifts")
    for n in N:
        nurse = active_nurses[n]
        nurse_max_night = nurse.get("maxNightShifts", max_night)
        terms = [is_night[(n, d)] for d in D if forced_label.get((n, d)) != "管夜"]
        if terms:
            model.add(sum(terms) <= nurse_max_night).only_enforce_if(a_max_night)

    # ── G. 夜勤 NG ペア ──
    a_ng_pair = make_assumption("nightNgPairs")
    for pair in night_ng_pairs:
        if len(pair) < 2:
            continue
        idx_a = next((i for i, nn in enumerate(active_nurses) if nn["id"] == pair[0]), None)
        idx_b = next((i for i, nn in enumerate(active_nurses) if nn["id"] == pair[1]), None)
        if idx_a is not None and idx_b is not None:
            for d in D:
                model.add_bool_or([a_ng_pair.negated(), is_night[(idx_a, d)].negated(), is_night[(idx_b, d)].negated()])

    # ── H. forced cells (希望/前月) ──
    a_forced = make_assumption("forcedCells")
    for (n, d), val in forced_shift.items():
        model.add(shifts[(n, d)] == val).only_enforce_if(a_forced)

    # ── I. forced 休/有 前日 NIGHT 禁止 ──
    a_off_prev_no_night = make_assumption("offPrevNoNight")
    for (n, d), lbl in forced_label.items():
        if lbl in ("休", "有") and d > 0 and (n, d - 1) not in forced_shift:
            model.add(is_night[(n, d - 1)] == 0).only_enforce_if(a_off_prev_no_night)

    # ── J. 夜勤人数 (緩和: ±1 許容) ──
    a_night_staff = make_assumption("nightStaff")
    for d in D:
        required = night_req_table[d]
        terms = [is_night[(n, d)] for n in N if forced_label.get((n, d)) != "管夜"]
        if not terms:
            continue
        total = sum(terms)
        model.add(total >= max(0, required - 1)).only_enforce_if(a_night_staff)
        model.add(total <= required + 1).only_enforce_if(a_night_staff)

    # ── K. 日勤人数 (緩和: -1 許容) ──
    a_day_staff = make_assumption("dayStaff")
    for d in D:
        required = weekend_day_staff if d in weekends else weekday_day_staff
        total_day = sum(is_day[(n, d)] for n in N)
        model.add(total_day >= max(0, required - 1)).only_enforce_if(a_day_staff)

    # assumption をモデルに登録して solve
    model.add_assumptions(list(assumptions.values()))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 8
    solver.parameters.num_workers = 4

    status = solver.solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return {"sufficientAssumptions": [], "note": "feasible under all assumptions (UNSAT analysis not applicable)"}
    if status != cp_model.INFEASIBLE:
        return {"sufficientAssumptions": [], "note": f"diagnostic solve status={solver.status_name(status)}"}

    sufficient_indices = solver.sufficient_assumptions_for_infeasibility()
    name_by_idx = {a.index: name for name, a in assumptions.items()}
    sufficient_names = [name_by_idx.get(idx, f"unknown_{idx}") for idx in sufficient_indices]
    return {
        "sufficientAssumptions": sufficient_names,
        "note": "UNSAT core: removing ALL of these (= relaxing those constraint groups) would make problem feasible",
    }


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

    # ── ハード制約: 夜→翌日OFF(明) / 夜→翌々日OFF(休) ──
    for n in N:
        for d in D:
            if d + 1 < num_days:
                model.add(is_off[(n, d + 1)] == 1).only_enforce_if(is_night[(n, d)])
            if d + 2 < num_days:
                model.add(is_off[(n, d + 2)] == 1).only_enforce_if(is_night[(n, d)])

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

    # 月末日の夜勤も通常通り許可。明は翌月初日、休は翌月2日目への申し送り
    # (post-proc では月内の明/休のみ生成、翌月分はフロントが prev_month_constraints
    #  として次月生成時に渡す前提)

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

    # ── 夜勤人数（ハード／relax_level=2で緩和）──
    # relax_level 0,1: 完全一致 (== required)
    # relax_level 2  : ±1 緩和 + ペナルティ
    night_dev_penalties = []
    for d in D:
        required = night_req_table[d]
        terms = [is_night[(n, d)] for n in N if forced_label.get((n, d)) != "管夜"]
        if not terms:
            continue
        total = sum(terms)
        if relax_level < 2:
            model.add(total == required)
        else:
            model.add(total >= max(0, required - 1))
            model.add(total <= required + 1)
            diff = model.new_int_var(-num_nurses, num_nurses, f"nd_{d}")
            model.add(diff == total - required)
            abs_diff = model.new_int_var(0, num_nurses, f"and_{d}")
            model.add_abs_equality(abs_diff, diff)
            night_dev_penalties.append((abs_diff, 5000))

    # ── 日勤人数（ハード／relax_level≥1で緩和）──
    # relax_level 0  : >= required
    # relax_level 1+ : >= required-1 + ペナルティ
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
    solver.parameters.max_time_in_seconds = 10
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


def _detect_suspicious_chains(active_nurses, requests, prev_month, num_days):
    """個別ナース内の "怪しい" 連続パターンを抽出。

    厳格 chain (夜→明→休) ルール下で実質的に矛盾するパターン:
      - day N 夜 → day N+2 明 (夜→明→明 の構造、休であるべき)
      - day N 夜 → day N+2 夜 (2連夜勤, 不可)
      - day N 明 → day N+1 夜 (連続 night chain の起点違反)
      - day N 明 → day N+1 明 (前日のチェーンが重なる)
      - day N 管夜 → day N+1 夜/日 等 (forced auto-fill 衝突)

    また月末日に「明」希望 → 翌月初日への "休" 申し送りが必要 (注意喚起のみ)
    """
    suspicious: list = []
    month_end_ake: list = []

    for n in active_nurses:
        nid = str(n["id"])
        nurse_reqs = requests.get(nid, {}) if isinstance(requests, dict) else {}
        nurse_prev = prev_month.get(nid, {}) if isinstance(prev_month, dict) else {}

        # Merge into a per-day map (request takes precedence over prev_month)
        locked: dict = {}
        for src_name, src in (("prev_month", nurse_prev), ("request", nurse_reqs)):
            if not isinstance(src, dict):
                continue
            for k, v in src.items():
                if str(k).startswith("_"):
                    continue
                try:
                    d = int(k) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= d < num_days:
                    locked[d] = {"label": v, "source": src_name}

        sorted_days = sorted(locked.keys())
        for d in sorted_days:
            lbl = locked[d]["label"]
            src = locked[d]["source"]

            # 夜 → d+2 が明/夜 (2連夜勤起こす形) は厳格 chain で違反
            if lbl == "夜" and d + 2 < num_days and (d + 2) in locked:
                nxt2 = locked[d + 2]["label"]
                if nxt2 == "夜":
                    suspicious.append({
                        "nurseId": nid,
                        "name": n.get("name", ""),
                        "pattern": f"day{d+1}=夜 → day{d+3}=夜",
                        "type": "double_night",
                        "message": (
                            f"2連夜勤 (夜明夜) はソルバー上不可: 夜→翌々日OFF が強制されるため。"
                            f" day{d+3}を休に変更するか day{d+1} の夜希望を削除してください"
                        ),
                    })
                elif nxt2 == "明":
                    suspicious.append({
                        "nurseId": nid,
                        "name": n.get("name", ""),
                        "pattern": f"day{d+1}=夜 → day{d+3}=明",
                        "type": "night_then_d2_ake",
                        "message": (
                            f"day{d+1}=夜なら day{d+3}は休のはず。明希望にすると前日が夜である必要があるが、"
                            f"day{d+2}は明 (= 夜の翌日) なので両立不可"
                        ),
                    })

            # 明 → d+1 が夜 は 2連夜勤を意味し strict chain で不可
            if lbl == "明" and d + 1 < num_days and (d + 1) in locked:
                nxt = locked[d + 1]["label"]
                if nxt == "夜":
                    suspicious.append({
                        "nurseId": nid,
                        "name": n.get("name", ""),
                        "pattern": f"day{d+1}=明 → day{d+2}=夜",
                        "type": "ake_then_night",
                        "message": (
                            f"明の翌日に夜は配置できない (= 2連夜勤の起点)。"
                            f" day{d+2}を休/有 に変更してください"
                        ),
                    })

            # 月末日が 明 → 翌月初日への 休 が必要 (注意喚起)
            if lbl == "明" and d == num_days - 1:
                month_end_ake.append({
                    "nurseId": nid,
                    "name": n.get("name", ""),
                    "day": d + 1,
                    "message": (
                        "月末日が明希望: 翌月初日が休になることをフロント側で確認してください"
                        " (当月のソルバーでは制約できません)"
                    ),
                })

    return suspicious, month_end_ake


def _detect_forced_conflicts(active_nurses, requests, prev_month, num_days):
    """forced cells (希望 + prev_month) の chain 整合性を検証。

    nightChain + forcedCells が UNSAT になる典型ケース:
      A) 夜希望の翌日に 夜/日/管夜/管明 等の非 OFF が固定
      B) 夜希望の翌々日に 日/管夜/管明 が固定 (休/有/夜/明 ならOK)
      C) 管夜希望の翌日が 管明 でない / 翌々日が休でない
      D) 明希望の前日が 夜 でない (prev_month の last day が夜なら例外)
      E) 同日に異なる希望/前月固定の重複
    """
    OFF_LABELS = {"休", "有", "明", "管明"}
    conflicts: list = []
    per_nurse_locked: list = []
    summary_by_nurse: dict = {}

    for n in active_nurses:
        nid = str(n["id"])
        # Collect forced cells per (day): {d: {"label", "source"}}
        locked: dict = {}

        # 1. requests (highest priority)
        nurse_reqs = requests.get(nid, {}) if isinstance(requests, dict) else {}
        if isinstance(nurse_reqs, dict):
            for day_str, label in nurse_reqs.items():
                try:
                    d = int(day_str) - 1
                except (TypeError, ValueError):
                    continue
                if not (0 <= d < num_days):
                    continue
                if d in locked and locked[d]["label"] != label:
                    conflicts.append({
                        "nurseId": nid,
                        "name": n.get("name", ""),
                        "day": d + 1,
                        "type": "duplicate_request",
                        "message": f"同日に異なる希望: {locked[d]['label']} vs {label}",
                    })
                locked[d] = {"label": label, "source": "request"}

        # 2. prev_month (skip if request already locked the same cell)
        nurse_prev = prev_month.get(nid, {}) if isinstance(prev_month, dict) else {}
        if isinstance(nurse_prev, dict):
            for key, label in nurse_prev.items():
                if str(key).startswith("_"):
                    continue
                try:
                    d = int(key) - 1
                except (TypeError, ValueError):
                    continue
                if not (0 <= d < num_days):
                    continue
                if d in locked:
                    if locked[d]["label"] != label:
                        conflicts.append({
                            "nurseId": nid,
                            "name": n.get("name", ""),
                            "day": d + 1,
                            "type": "request_vs_prev_conflict",
                            "message": f"希望({locked[d]['label']}) と 前月({label}) で固定値が異なる",
                        })
                    continue
                locked[d] = {"label": label, "source": "prev_month"}

        if not locked:
            continue

        # 3. Chain consistency check (in date order)
        sorted_days = sorted(locked.keys())
        for d in sorted_days:
            lbl = locked[d]["label"]
            src = locked[d]["source"]

            # Case A/B: 夜希望 (現在のソルバーは strict chain: 夜→明→休 を強制)
            if lbl == "夜":
                # day d+1: 明/休/有 のみ (= OFF)。日/夜/管夜/管明 は不可
                if d + 1 < num_days and (d + 1) in locked:
                    nxt = locked[d + 1]["label"]
                    if nxt not in ("明", "休", "有"):
                        conflicts.append({
                            "nurseId": nid,
                            "name": n.get("name", ""),
                            "day": d + 1,
                            "type": "night_then_invalid_d1",
                            "message": (
                                f"day{d+1}=夜({src}) → day{d+2}={nxt}({locked[d+1]['source']}) 固定。"
                                f" 明/休/有 のいずれかであるべき"
                            ),
                        })
                # day d+2: 休/有/明 のみ (= OFF)。夜 は 2連夜勤になるため不可。日 も不可
                if d + 2 < num_days and (d + 2) in locked:
                    nxt2 = locked[d + 2]["label"]
                    if nxt2 not in ("休", "有", "明"):
                        conflicts.append({
                            "nurseId": nid,
                            "name": n.get("name", ""),
                            "day": d + 1,
                            "type": "night_then_invalid_d2",
                            "message": (
                                f"day{d+1}=夜({src}) → day{d+3}={nxt2}({locked[d+2]['source']}) 固定。"
                                f" 休/有/明 のいずれかであるべき (2連夜勤=夜明夜は不可)"
                            ),
                        })

            # Case C: 管夜希望
            elif lbl == "管夜":
                if d + 1 < num_days and (d + 1) in locked and locked[d + 1]["label"] != "管明":
                    conflicts.append({
                        "nurseId": nid,
                        "name": n.get("name", ""),
                        "day": d + 1,
                        "type": "kannight_invalid_d1",
                        "message": f"day{d+1}=管夜({src}) → day{d+2}={locked[d+1]['label']} 固定。管明であるべき",
                    })
                if d + 2 < num_days and (d + 2) in locked and locked[d + 2]["label"] not in ("休", "有"):
                    conflicts.append({
                        "nurseId": nid,
                        "name": n.get("name", ""),
                        "day": d + 1,
                        "type": "kannight_invalid_d2",
                        "message": f"day{d+1}=管夜({src}) → day{d+3}={locked[d+2]['label']} 固定。休/有 のいずれかであるべき",
                    })

            # Case D: 明希望 → 前日は 夜 のはず
            elif lbl == "明":
                if d > 0:
                    if (d - 1) in locked:
                        prv = locked[d - 1]["label"]
                        if prv != "夜":
                            conflicts.append({
                                "nurseId": nid,
                                "name": n.get("name", ""),
                                "day": d + 1,
                                "type": "ake_no_prev_night",
                                "message": f"day{d+1}=明({src}) だが前日 day{d}={prv}({locked[d-1]['source']}) 固定。夜であるべき",
                            })
                # 明 の翌日: 休/有 (strict chain) のみ。夜 は 2連夜勤になるため不可
                if d + 1 < num_days and (d + 1) in locked:
                    nxt = locked[d + 1]["label"]
                    if nxt not in ("休", "有"):
                        conflicts.append({
                            "nurseId": nid,
                            "name": n.get("name", ""),
                            "day": d + 1,
                            "type": "ake_then_invalid",
                            "message": (
                                f"day{d+1}=明({src}) → day{d+2}={nxt}({locked[d+1]['source']}) 固定。"
                                f" 休/有 のいずれかであるべき (明→夜 の 2連夜勤は不可)"
                            ),
                        })

            # Case E: 管明希望 → 前日は 管夜 のはず
            elif lbl == "管明":
                if d > 0 and (d - 1) in locked and locked[d - 1]["label"] != "管夜":
                    conflicts.append({
                        "nurseId": nid,
                        "name": n.get("name", ""),
                        "day": d + 1,
                        "type": "kanake_no_prev_kannight",
                        "message": f"day{d+1}=管明({src}) だが前日 day{d}={locked[d-1]['label']} 固定。管夜であるべき",
                    })

        per_nurse_locked.append({
            "id": n["id"],
            "name": n.get("name", ""),
            "lockedCount": len(locked),
            "lockedDays": [
                {"day": d + 1, "shift": locked[d]["label"], "source": locked[d]["source"]}
                for d in sorted_days
            ],
        })
        summary_by_nurse[nid] = locked

    # 矛盾のあるナースのみで perNurseLockedDays を絞る (情報量制御)
    conflict_nurse_ids = {c["nurseId"] for c in conflicts}
    locked_with_conflict = [
        x for x in per_nurse_locked if str(x["id"]) in conflict_nurse_ids
    ]

    return conflicts, per_nurse_locked, locked_with_conflict


def _per_nurse_summary(active_nurses, max_night_default, requests, prev_month):
    """個別ナースのケイパビリティと希望サマリ。INFEASIBLE 解析の手がかり用。"""
    summary = []
    position_stats: dict = {}
    max_buckets: dict = {}

    for n in active_nurses:
        nid = str(n["id"])
        no_night = bool(n.get("noNightShift", False))
        no_day = bool(n.get("noDayShift", False))
        max_night = n.get("maxNightShifts", max_night_default)
        position = n.get("position", "未設定")

        nurse_reqs = requests.get(nid, {}) if isinstance(requests, dict) else {}
        prev_reqs = prev_month.get(nid, {}) if isinstance(prev_month, dict) else {}
        request_off_count = sum(1 for v in nurse_reqs.values() if v in ("休", "有"))
        request_night_count = sum(1 for v in nurse_reqs.values() if v in ("夜", "管夜"))
        prev_consec = prev_reqs.get("_consecDays", 0) if isinstance(prev_reqs, dict) else 0
        prev_locked = sum(1 for k in prev_reqs.keys() if not str(k).startswith("_")) \
                      if isinstance(prev_reqs, dict) else 0

        summary.append({
            "id": n["id"],
            "name": n.get("name", ""),
            "position": position,
            "noNightShift": no_night,
            "noDayShift": no_day,
            "maxNightShifts": max_night,
            "requestsOff": request_off_count,
            "requestsNight": request_night_count,
            "prevConsecDays": prev_consec,
            "prevLockedCells": prev_locked,
        })

        if position not in position_stats:
            position_stats[position] = {"count": 0, "nightCapable": 0,
                                         "dayCapable": 0, "totalMaxNights": 0}
        ps = position_stats[position]
        ps["count"] += 1
        if not no_night:
            ps["nightCapable"] += 1
            ps["totalMaxNights"] += max_night
        if not no_day:
            ps["dayCapable"] += 1

        bucket_key = f"max={max_night}" if not no_night else "noNight"
        max_buckets[bucket_key] = max_buckets.get(bucket_key, 0) + 1

    return summary, position_stats, max_buckets


def _request_distribution(requests, active_nurses, num_days, weekday_day_staff,
                           weekend_day_staff, weekends, night_req_table):
    """希望休/夜勤の日別分布。要件を満たせる人数が残るかチェック。"""
    if not isinstance(requests, dict):
        return {
            "totalRequests": 0,
            "perDayMaxRequests": [],
            "daysWithExcessiveRequests": [],
            "dailyRequestSummary": [],
        }

    nurse_id_set = {str(n["id"]) for n in active_nurses}
    total_active = len(active_nurses)

    per_day_off = [0] * num_days       # 休/有 のみ (純粋な休み希望)
    per_day_ake = [0] * num_days       # 明
    per_day_kanake = [0] * num_days    # 管明
    per_day_night = [0] * num_days     # 夜
    per_day_kannight = [0] * num_days  # 管夜
    per_day_day = [0] * num_days       # 日
    total_requests = 0

    for nid, req_map in requests.items():
        if str(nid) not in nurse_id_set or not isinstance(req_map, dict):
            continue
        for day_str, label in req_map.items():
            try:
                d = int(day_str) - 1
            except (TypeError, ValueError):
                continue
            if not (0 <= d < num_days):
                continue
            total_requests += 1
            if label in ("休", "有"):
                per_day_off[d] += 1
            elif label == "明":
                per_day_ake[d] += 1
            elif label == "管明":
                per_day_kanake[d] += 1
            elif label == "夜":
                per_day_night[d] += 1
            elif label == "管夜":
                per_day_kannight[d] += 1
            elif label == "日":
                per_day_day[d] += 1

    # 希望が多い日 (上位5件)
    sorted_days = sorted(
        [(d, per_day_off[d]) for d in range(num_days)],
        key=lambda x: -x[1],
    )[:5]
    per_day_max_requests = [{"day": d + 1, "offRequests": cnt} for d, cnt in sorted_days if cnt > 0]

    # 希望休が多すぎて要件を満たせない日を検出
    excessive: list = []
    daily_summary: list = []
    for d in range(num_days):
        off_req = per_day_off[d]
        ake_req = per_day_ake[d]
        kanake_req = per_day_kanake[d]
        night_req_user = per_day_night[d]
        kannight_req = per_day_kannight[d]
        day_req_user = per_day_day[d]
        night_req = night_req_table[d]
        day_req = weekend_day_staff if d in weekends else weekday_day_staff

        # この日に勤務可能 (forced 休以外) なナース数
        # 明・管明 も勤務にはカウントされないので除外
        forced_off_total = off_req + ake_req + kanake_req
        feasible_workers = total_active - forced_off_total
        needed = night_req + day_req

        issues: list[str] = []
        if feasible_workers < needed:
            issues.append(f"勤務可能{feasible_workers}<必要{needed}")
            excessive.append({
                "day": d + 1,
                "offRequests": off_req,
                "akeRequests": ake_req,
                "kanakeRequests": kanake_req,
                "nightReq": night_req,
                "dayReq": day_req,
                "neededWorkers": needed,
                "feasibleWorkers": feasible_workers,
                "shortfall": needed - feasible_workers,
            })
        if night_req_user > night_req:
            issues.append(f"夜希望{night_req_user}>必要{night_req}")
        if day_req_user + night_req_user + off_req + ake_req + kanake_req + kannight_req > 0:
            daily_summary.append({
                "day": d + 1,
                "isWeekend": d in weekends,
                "off": off_req,
                "ake": ake_req,
                "kanake": kanake_req,
                "night": night_req_user,
                "kannight": kannight_req,
                "dayShift": day_req_user,
                "nightReq": night_req,
                "dayReq": day_req,
                "feasibleWorkers": feasible_workers,
                "issues": issues,
            })

    return {
        "totalRequests": total_requests,
        "perDayMaxRequests": per_day_max_requests,
        "dailyRequestSummary": daily_summary,
        "daysWithExcessiveRequests": excessive,
    }


def _preflight_diagnostics(active_nurses, night_req_table, weekday_day_staff,
                            weekend_day_staff, weekends, num_days, max_night_default,
                            max_consec=3, max_days_off=10,
                            prev_month=None, requests=None):
    """生成リクエストが数学的に成立し得るかを事前にチェックする。"""
    warnings: list[str] = []

    # 0. パース済み入力の echo (フロントの送信値と diagnostics を突き合わせるため)
    weekend_day_count = sum(1 for d in range(num_days) if d in weekends)
    weekday_day_count = num_days - weekend_day_count
    parsed_config = {
        "weekdayDayStaff": weekday_day_staff,
        "weekendDayStaff": weekend_day_staff,
        "maxNightShiftsDefault": max_night_default,
        "maxConsecutiveDays": max_consec,
        "maxDaysOff": max_days_off,
        "numDays": num_days,
        "weekendCount": weekend_day_count,  # weekends ∩ [0, num_days)
        "weekdayCount": weekday_day_count,
    }

    # 1. 夜勤総量
    night_demand = sum(night_req_table)
    night_capable = [n for n in active_nurses if not n.get("noNightShift", False)]
    night_capacity = sum(n.get("maxNightShifts", max_night_default) for n in night_capable)
    if night_capacity < night_demand:
        warnings.append(
            f"夜勤総量不足: capacity={night_capacity} < demand={night_demand} "
            f"(夜勤可能{len(night_capable)}名 × 個人上限平均{night_capacity/max(len(night_capable),1):.1f})"
        )

    # 2. 日勤総量
    day_demand = (weekday_day_count * weekday_day_staff +
                  weekend_day_count * weekend_day_staff)
    day_capable = [n for n in active_nurses if not n.get("noDayShift", False)]
    # 日勤の物理上限: 各ナースの 1ヶ月の最大勤務セル数 × 人数
    # max_consec 制約から 1人あたり最大ワークセル数 ≈ ceil(num_days * mc / (mc+1))
    per_nurse_max_work = (num_days * max_consec + max_consec) // (max_consec + 1)
    day_capacity_strict = len(day_capable) * per_nurse_max_work
    if day_capacity_strict < day_demand:
        warnings.append(
            f"日勤総量不足: capacity={day_capacity_strict} < demand={day_demand} "
            f"(日勤可能{len(day_capable)}名 × 連続上限{max_consec}に基づく1人最大{per_nurse_max_work}日)"
        )

    # 3. 夜勤可能ナースが0
    if not night_capable:
        warnings.append("夜勤可能なナースがいません (全員 noNightShift=true)")

    # 4. 日勤可能ナースが0
    if not day_capable:
        warnings.append("日勤可能なナースがいません (全員 noDayShift=true)")

    # 5. 日勤要件が異常に低い (フロントの設定ミスを検出)
    if weekday_day_staff <= 1 or weekend_day_staff <= 1:
        warnings.append(
            f"日勤必要人数が異常に低い: weekday={weekday_day_staff}, weekend={weekend_day_staff}"
            f" — フロント設定値を確認してください"
        )

    # 6. weekends が num_days を超えている / 範囲外を含む
    out_of_range = [d for d in weekends if not (0 <= d < num_days)]
    if out_of_range:
        warnings.append(
            f"weekends に範囲外のインデックス {out_of_range[:5]}{'…' if len(out_of_range)>5 else ''} が含まれます"
            f" (許容: 0〜{num_days-1})"
        )

    # 7. prev_month_constraints の orphan ID 検出
    nurse_ids = {str(n["id"]) for n in active_nurses}
    orphan_prev_ids: list[str] = []
    if prev_month:
        prev_keys = {str(k) for k in prev_month.keys()}
        orphan_prev_ids = sorted(prev_keys - nurse_ids)
        if orphan_prev_ids:
            warnings.append(
                f"prevMonthConstraints に存在しないナースID: "
                f"{orphan_prev_ids[:5]}{'…' if len(orphan_prev_ids)>5 else ''} "
                f"({len(orphan_prev_ids)}件) — ソルバーは無視して進めます"
            )

    # 8. requests の orphan ID 検出
    orphan_req_ids: list[str] = []
    if requests:
        req_keys = {str(k) for k in requests.keys()}
        orphan_req_ids = sorted(req_keys - nurse_ids)
        if orphan_req_ids:
            warnings.append(
                f"requests に存在しないナースID: "
                f"{orphan_req_ids[:5]}{'…' if len(orphan_req_ids)>5 else ''} "
                f"({len(orphan_req_ids)}件) — ソルバーは無視して進めます"
            )

    # 個別ナースサマリ・希望分布
    nurse_summary, position_stats, max_buckets = _per_nurse_summary(
        active_nurses, max_night_default, requests or {}, prev_month or {}
    )
    request_dist = _request_distribution(
        requests or {}, active_nurses, num_days, weekday_day_staff,
        weekend_day_staff, weekends, night_req_table
    )

    # 希望が多すぎて要件を満たせない日があれば warning
    if request_dist["daysWithExcessiveRequests"]:
        days_str = ", ".join(
            f"Day{x['day']}(休望{x['offRequests']}/必要{x['neededWorkers']})"
            for x in request_dist["daysWithExcessiveRequests"][:3]
        )
        warnings.append(
            f"希望休が必要人数を圧迫: {days_str}"
            f" (詳細: requestStats.daysWithExcessiveRequests)"
        )

    # forced cells の chain 整合性
    forced_conflicts, per_nurse_locked, per_nurse_locked_conflict = \
        _detect_forced_conflicts(active_nurses, requests or {}, prev_month or {}, num_days)
    if forced_conflicts:
        sample = forced_conflicts[:3]
        msg = "; ".join(c["message"] for c in sample)
        warnings.append(
            f"希望/前月固定の {len(forced_conflicts)} 件で chain 矛盾: {msg}"
            f"{' …' if len(forced_conflicts) > 3 else ''} (詳細: forcedCellConflicts)"
        )

    # 怪しい連続パターン (個人内) と月末日 明
    suspicious_chains, month_end_ake = _detect_suspicious_chains(
        active_nurses, requests or {}, prev_month or {}, num_days
    )
    if suspicious_chains:
        sample = suspicious_chains[:3]
        msg = "; ".join(f"{c['name']} {c['pattern']}" for c in sample)
        warnings.append(
            f"怪しい連続希望 {len(suspicious_chains)} 件: {msg}"
            f"{' …' if len(suspicious_chains) > 3 else ''} (詳細: suspiciousChainRequests)"
        )
    if month_end_ake:
        names = ", ".join(x["name"] for x in month_end_ake[:3])
        warnings.append(
            f"月末日が明希望のナース {len(month_end_ake)}名: {names}"
            f" — 翌月初日の休がフロントで適切に申し送られているか確認 (詳細: monthEndAke)"
        )

    # chain rule の明文化 (フロントに現在のルールを伝える)
    chain_rule = {
        "夜→翌日": "OFF (= 明/休/有 のいずれか)。ハード制約",
        "夜→翌々日": "OFF (= 明/休/有 のいずれか)。ハード制約。2連夜勤(夜明夜)は不可",
        "明→翌日": "休/有 (validator)。明→夜 は 2連夜勤になるため不可",
        "管夜→翌日": "管明 (forced auto-fill)",
        "管夜→翌々日": "休 (forced auto-fill)",
        "管明→翌日": "休/有 (validator)",
        "連続勤務カウント": "夜・日 は working、明・管明・休・有 は non-working",
        "月末": "月末日も nightShiftPattern を適用。明/休 は翌月へキャリーオーバー",
    }

    return {
        "parsedConfig": parsed_config,
        "nightDemand": night_demand,
        "nightCapacity": night_capacity,
        "nightCapableCount": len(night_capable),
        "dayDemand": day_demand,
        "dayCapacity": day_capacity_strict,
        "dayCapableCount": len(day_capable),
        "orphanPrevIds": orphan_prev_ids,
        "orphanRequestIds": orphan_req_ids,
        "perPositionStats": position_stats,
        "nightCapableByMaxShifts": max_buckets,
        "individualNurseSummary": nurse_summary,
        "requestStats": request_dist,
        "forcedCellConflicts": forced_conflicts,
        "suspiciousChainRequests": suspicious_chains,
        "monthEndAke": month_end_ake,
        "chainRule": chain_rule,
        "perNurseLockedDays": per_nurse_locked,
        "perNurseLockedDaysConflictOnly": per_nurse_locked_conflict,
        "perDayDemandSample": [
            {"day": d + 1,
             "isWeekend": d in weekends,
             "dayReq": weekend_day_staff if d in weekends else weekday_day_staff,
             "nightReq": night_req_table[d]}
            for d in list(range(min(7, num_days))) + list(range(max(0, num_days-3), num_days))
        ],
        "warnings": warnings,
    }


def _validate(data, params, relax_level=0):
    """構造エラーは常にチェック。人数要件は relax_level に応じて緩める。"""
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

    # 夜勤人数: relax_level<2 で完全一致要求、relax_level>=2 で ±1 許容
    for d in range(num_days):
        actual = sum(1 for sh in data.values() if sh[d] == "夜")
        req = night_req_table[d]
        if relax_level < 2:
            if actual != req:
                errors.append(f"Day {d+1}: 夜勤人数 {actual} ≠ 要件 {req}")
        else:
            if abs(actual - req) > 1:
                errors.append(f"Day {d+1}: 夜勤人数 {actual} (要件 {req}, 許容±1超過)")

    # 日勤人数: relax_level==0 で完全要求、relax_level>=1 で -1 許容
    for d in range(num_days):
        actual = sum(1 for sh in data.values() if sh[d] == "日")
        req = weekend_day_staff if d in weekends else weekday_day_staff
        threshold = req if relax_level == 0 else max(0, req - 1)
        if actual < threshold:
            errors.append(f"Day {d+1}: 日勤人数 {actual} < 要件 {threshold}")

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
    # ── FULL REQUEST DUMP (Cloud Run logs で原因追跡用) ──
    try:
        import json as _json
        dump = _json.dumps(request_data, ensure_ascii=False, default=str)
        if len(dump) > 8000:
            dump = dump[:8000] + f"...[truncated, full length={len(dump)}]"
        _log(f"FULL REQUEST DUMP: {dump}")
    except Exception as e:
        _log(f"REQUEST DUMP failed: {e}")

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
    # 月末日も nightShiftPattern を適用（明/休は翌月へのキャリーオーバー）
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

    # ── 事前 feasibility 診断 ──
    diagnostics = _preflight_diagnostics(
        active_nurses, night_req_table, weekday_day_staff, weekend_day_staff,
        weekends_combined, num_days, max_night,
        max_consec=max_consec, max_days_off=max_days_off,
        prev_month=prev_month, requests=requests,
    )
    _log(f"PREFLIGHT: parsedConfig={diagnostics['parsedConfig']}")
    _log(f"PREFLIGHT: nightDemand={diagnostics['nightDemand']} "
         f"nightCapacity={diagnostics['nightCapacity']} "
         f"dayDemand={diagnostics['dayDemand']} "
         f"dayCapacity={diagnostics['dayCapacity']}")
    if diagnostics["warnings"]:
        for w in diagnostics["warnings"]:
            _log(f"PREFLIGHT WARNING: {w}")

    forbidden_solutions: list[dict] = []
    results: list[dict] = []
    pattern_labels = ["パターンA", "パターンB", "パターンC", "パターンD", "パターンE"]

    for pat_idx in range(num_patterns):
        label = pattern_labels[pat_idx] if pat_idx < len(pattern_labels) else f"パターン{pat_idx + 1}"
        _log(f"=== {label} 開始 (pat_idx={pat_idx}) ===")

        chosen = None
        chosen_errors: list[str] = []
        last_status = None
        attempts: list[dict] = []  # 各 relax_level の試行記録

        # 試行: relax 0(完全遵守) → 1(日勤緩和) → 2(日勤・夜勤緩和)
        for relax_level in (0, 1, 2):
            t0 = time.time()
            res = _solve_one_pattern(params, forbidden_solutions, relax_level=relax_level)
            elapsed = time.time() - t0
            last_status = res["status"]
            attempts.append({
                "relaxLevel": relax_level,
                "status": last_status,
                "elapsedSec": round(elapsed, 2),
            })
            _log(f"  {label} relax={relax_level}: status={last_status} elapsed={elapsed:.2f}s")

            if res["raw"] is None:
                continue  # この緩和では解なし、次の緩和へ
            final_data = _post_process(res["raw"], active_nurses, forced_label, num_days)
            errors = _validate(final_data, params, relax_level=relax_level)
            if not errors:
                attempts[-1]["validationErrors"] = 0
                chosen = {
                    "raw": res["raw"],
                    "data": final_data,
                    "objective": res["objective"],
                    "relax_level": relax_level,
                }
                _log(f"  {label} relax={relax_level}: 採用")
                break
            attempts[-1]["validationErrors"] = len(errors)
            _log(f"  {label} relax={relax_level}: validation NG ({len(errors)}件) → 次の緩和へ")
            chosen_errors = errors

        if chosen is None:
            err_msg = f"解が見つかりませんでした (status={last_status})"
            if diagnostics["warnings"]:
                err_msg += " — " + "; ".join(diagnostics["warnings"][:2])
            _log(f"!!! {label} 全 relax_level で失敗: {err_msg}")

            # UNSAT core 解析 (最初のパターンの失敗時のみ実施: 結果は同条件で同じ)
            unsat_info = None
            if pat_idx == 0:
                _log(f"!!! UNSAT core 解析を開始 (assumption-based)")
                t_unsat = time.time()
                try:
                    unsat_info = _diagnose_infeasible(params)
                    _log(f"!!! UNSAT core: {unsat_info.get('sufficientAssumptions')} ({time.time()-t_unsat:.2f}s)")
                except Exception as e:
                    _log(f"!!! UNSAT core 解析失敗: {e}")
                    unsat_info = {"error": str(e)}

            results.append({
                "label": label,
                "data": {},
                "score": 0,
                "metrics": {
                    "solverUsed": True,
                    "error": err_msg,
                    "solverStatus": last_status,
                    "validationErrors": chosen_errors[:10],
                    "attempts": attempts,
                    "diagnostics": diagnostics,
                    "unsatCore": unsat_info,
                    # フロント描画が undefined で落ちないように 0 埋め
                    "relaxLevel": -1,
                    "nightBalance": 0,
                    "dayShortage": 0,
                    "nightShortage": 0,
                    "consecViolations": 0,
                    "requestMatch": 0,
                    "avgDaysOff": 0,
                    "nullCells": 0,
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
                "solverUsed": True,
                "relaxLevel": chosen["relax_level"],
                "nightBalance": round(night_balance, 1),
                "dayShortage": total_day_shortage,
                "nightShortage": total_night_shortage,
                "consecViolations": consec_violations,
                "requestMatch": round(req_match, 1),
                "avgDaysOff": round(avg_off, 1),
                "nullCells": null_cells,
                "attempts": attempts,
                "diagnostics": diagnostics,
            },
        })

    return results
