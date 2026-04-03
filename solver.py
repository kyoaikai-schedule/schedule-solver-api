"""OR-Tools CP-SAT nurse schedule solver."""

from ortools.sat.python import cp_model

# Shift constants
NULL = 0
DAY = 1
NIGHT = 2
AKE = 3       # 明け
KAN_NIGHT = 4 # 管夜
KAN_AKE = 5   # 管明
OFF = 6       # 休

SHIFT_LABELS = {NULL: "", DAY: "日", NIGHT: "夜", AKE: "明", KAN_NIGHT: "管夜", KAN_AKE: "管明", OFF: "休"}

REQUEST_MAP = {"日": DAY, "夜": NIGHT, "明": AKE, "管夜": KAN_NIGHT, "管明": KAN_AKE, "休": OFF, "有": OFF}
HARD_REQUEST = {"休", "有"}


def solve_schedule(request_data: dict) -> list[dict]:
    nurses = request_data["nurses"]
    days_in_month = request_data["daysInMonth"]
    config = request_data["config"]
    requests = request_data.get("requests", {})
    night_ng_pairs = request_data.get("nightNgPairs", [])
    prev_month = request_data.get("prevMonthConstraints", {})
    holidays = set(request_data.get("holidays", []))
    weekends = set(request_data.get("weekends", []))
    num_patterns = request_data.get("numPatterns", 3)

    weekday_day_staff = config["weekdayDayStaff"]
    weekend_day_staff = config["weekendDayStaff"]
    night_pattern = config["nightShiftPattern"]
    max_night = config.get("maxNightShifts", 6)
    max_days_off = config.get("maxDaysOff", 10)
    max_consec = config.get("maxConsecutiveDays", 3)
    max_double_night = config.get("maxDoubleNightPairs", 2)

    # Filter out excluded nurses
    active_nurses = [n for n in nurses if not n.get("excludeFromGeneration", False)]
    num_nurses = len(active_nurses)
    num_days = days_in_month
    D = range(num_days)
    N = range(num_nurses)

    forbidden_solutions: list[dict] = []
    results = []
    pattern_labels = ["パターンA", "パターンB", "パターンC", "パターンD", "パターンE"]

    for pat_idx in range(num_patterns):
        model = cp_model.CpModel()

        # --- Variables ---
        shifts = {}
        is_day = {}
        is_night = {}
        is_ake = {}
        is_kan_night = {}
        is_kan_ake = {}
        is_off = {}
        is_working = {}

        for n in N:
            for d in D:
                shifts[(n, d)] = model.NewIntVar(0, 6, f"shift_n{n}_d{d}")
                is_day[(n, d)] = model.NewBoolVar(f"is_day_n{n}_d{d}")
                is_night[(n, d)] = model.NewBoolVar(f"is_night_n{n}_d{d}")
                is_ake[(n, d)] = model.NewBoolVar(f"is_ake_n{n}_d{d}")
                is_kan_night[(n, d)] = model.NewBoolVar(f"is_kan_night_n{n}_d{d}")
                is_kan_ake[(n, d)] = model.NewBoolVar(f"is_kan_ake_n{n}_d{d}")
                is_off[(n, d)] = model.NewBoolVar(f"is_off_n{n}_d{d}")
                is_working[(n, d)] = model.NewBoolVar(f"is_working_n{n}_d{d}")

                # Link bool vars to shift var
                model.Add(shifts[(n, d)] == DAY).OnlyEnforceIf(is_day[(n, d)])
                model.Add(shifts[(n, d)] != DAY).OnlyEnforceIf(is_day[(n, d)].Not())
                model.Add(shifts[(n, d)] == NIGHT).OnlyEnforceIf(is_night[(n, d)])
                model.Add(shifts[(n, d)] != NIGHT).OnlyEnforceIf(is_night[(n, d)].Not())
                model.Add(shifts[(n, d)] == AKE).OnlyEnforceIf(is_ake[(n, d)])
                model.Add(shifts[(n, d)] != AKE).OnlyEnforceIf(is_ake[(n, d)].Not())
                model.Add(shifts[(n, d)] == KAN_NIGHT).OnlyEnforceIf(is_kan_night[(n, d)])
                model.Add(shifts[(n, d)] != KAN_NIGHT).OnlyEnforceIf(is_kan_night[(n, d)].Not())
                model.Add(shifts[(n, d)] == KAN_AKE).OnlyEnforceIf(is_kan_ake[(n, d)])
                model.Add(shifts[(n, d)] != KAN_AKE).OnlyEnforceIf(is_kan_ake[(n, d)].Not())
                model.Add(shifts[(n, d)] == OFF).OnlyEnforceIf(is_off[(n, d)])
                model.Add(shifts[(n, d)] != OFF).OnlyEnforceIf(is_off[(n, d)].Not())

                # Working = day or night or kan_night
                model.AddBoolOr([is_day[(n, d)], is_night[(n, d)], is_kan_night[(n, d)]]).OnlyEnforceIf(is_working[(n, d)])
                model.AddBoolAnd([is_day[(n, d)].Not(), is_night[(n, d)].Not(), is_kan_night[(n, d)].Not()]).OnlyEnforceIf(is_working[(n, d)].Not())

        # --- Hard Constraints ---

        for n in N:
            nurse = active_nurses[n]
            nurse_id = str(nurse["id"])
            nurse_requests = requests.get(nurse_id, {})

            for d in D:
                # H1: Night -> next day is Ake
                if d < num_days - 1:
                    model.Add(shifts[(n, d + 1)] == AKE).OnlyEnforceIf(is_night[(n, d)])
                    model.Add(shifts[(n, d + 1)] == KAN_AKE).OnlyEnforceIf(is_kan_night[(n, d)])

                # H2: Ake -> next day is Off
                if d < num_days - 1:
                    model.Add(shifts[(n, d + 1)] == OFF).OnlyEnforceIf(is_ake[(n, d)])
                    model.Add(shifts[(n, d + 1)] == OFF).OnlyEnforceIf(is_kan_ake[(n, d)])

            # H3: Consecutive working days limit
            for d in D:
                window = max_consec + 1
                if d + window <= num_days:
                    model.Add(sum(is_working[(n, d + k)] for k in range(window)) <= max_consec)

            # H3 extended: account for previous month consecutive days
            prev_nurse = prev_month.get(nurse_id, {})
            prev_consec = prev_nurse.get("_consecDays", 0)
            if prev_consec > 0:
                remaining = max_consec - prev_consec
                if remaining <= 0:
                    # Must not work on day 0
                    model.Add(is_working[(n, 0)] == 0)
                else:
                    # Limit first `remaining+1` days
                    for end_d in range(1, min(remaining + 1, num_days)):
                        model.Add(
                            sum(is_working[(n, k)] for k in range(end_d + 1)) <= remaining
                        )

            # H4: No 3 consecutive night shifts (夜明夜明夜 pattern forbidden)
            for d in range(num_days - 4):
                b = model.NewBoolVar(f"triple_night_n{n}_d{d}")
                model.AddBoolAnd([is_night[(n, d)], is_night[(n, d + 2)], is_night[(n, d + 4)]]).OnlyEnforceIf(b)
                model.Add(b == 0)

            # H6: No night shift restriction
            if nurse.get("noNightShift", False):
                for d in D:
                    model.Add(is_night[(n, d)] == 0)
                    model.Add(is_kan_night[(n, d)] == 0)

            # H7: No day shift restriction
            if nurse.get("noDayShift", False):
                for d in D:
                    model.Add(is_day[(n, d)] == 0)

            # H9: Hard request constraints (休, 有)
            for day_str, req_type in nurse_requests.items():
                day_idx = int(day_str) - 1
                if 0 <= day_idx < num_days and req_type in HARD_REQUEST:
                    model.Add(shifts[(n, day_idx)] == OFF)

        # H5: Night NG pairs
        for pair in night_ng_pairs:
            id_a, id_b = pair[0], pair[1]
            idx_a = next((i for i, nn in enumerate(active_nurses) if nn["id"] == id_a), None)
            idx_b = next((i for i, nn in enumerate(active_nurses) if nn["id"] == id_b), None)
            if idx_a is not None and idx_b is not None:
                for d in D:
                    model.AddBoolOr([is_night[(idx_a, d)].Not(), is_night[(idx_b, d)].Not()])

        # H8: Previous month fixed shifts
        for nurse_id_str, prev_data in prev_month.items():
            n_idx = next((i for i, nn in enumerate(active_nurses) if str(nn["id"]) == nurse_id_str), None)
            if n_idx is None:
                continue
            for key, val in prev_data.items():
                if key.startswith("_"):
                    continue
                # key is a day number (1-indexed), val is the shift type
                day_idx = int(key) - 1
                if 0 <= day_idx < num_days and val in REQUEST_MAP:
                    model.Add(shifts[(n_idx, day_idx)] == REQUEST_MAP[val])

        # --- Soft Constraints (Penalties) ---
        penalties = []

        # S1: Night shift staffing per day
        for d in D:
            pattern_idx = d % len(night_pattern)
            required_night = night_pattern[pattern_idx]
            total_night = sum(is_night[(n, d)] for n in N)
            over = model.NewIntVar(0, num_nurses, f"night_over_d{d}")
            under = model.NewIntVar(0, num_nurses, f"night_under_d{d}")
            model.Add(total_night - required_night == over - under)
            penalties.append((over, 1000))
            penalties.append((under, 1000))

        # S2: Day shift staffing per day
        for d in D:
            required_day = weekend_day_staff if d in weekends else weekday_day_staff
            total_day = sum(is_day[(n, d)] for n in N)
            over = model.NewIntVar(0, num_nurses, f"day_over_d{d}")
            under = model.NewIntVar(0, num_nurses, f"day_under_d{d}")
            model.Add(total_day - required_day == over - under)
            penalties.append((over, 500))
            penalties.append((under, 500))

        # S3: Night shift balance across nurses
        night_counts = []
        for n in N:
            nurse = active_nurses[n]
            nc = model.NewIntVar(0, num_days, f"night_count_n{n}")
            model.Add(nc == sum(is_night[(n, d)] for d in D))
            night_counts.append(nc)

        if len(night_counts) > 1:
            max_nc = model.NewIntVar(0, num_days, "max_night_count")
            min_nc = model.NewIntVar(0, num_days, "min_night_count")
            model.AddMaxEquality(max_nc, night_counts)
            model.AddMinEquality(min_nc, night_counts)
            night_diff = model.NewIntVar(0, num_days, "night_diff")
            model.Add(night_diff == max_nc - min_nc)
            penalties.append((night_diff, 300))

        # S4: Night shift distribution within month (spread out)
        for n in N:
            # Divide month into 3 segments and balance night counts
            seg_len = num_days // 3
            segs = []
            for s in range(3):
                start = s * seg_len
                end = (s + 1) * seg_len if s < 2 else num_days
                seg_count = model.NewIntVar(0, num_days, f"seg_night_n{n}_s{s}")
                model.Add(seg_count == sum(is_night[(n, d)] for d in range(start, end)))
                segs.append(seg_count)
            for i in range(len(segs)):
                for j in range(i + 1, len(segs)):
                    diff = model.NewIntVar(0, num_days, f"seg_diff_n{n}_{i}_{j}")
                    model.AddAbsEquality(diff, segs[i] - segs[j])
                    penalties.append((diff, 50))

        # S5: Double night pair limit
        for n in N:
            double_night_count = model.NewIntVar(0, num_days, f"double_night_n{n}")
            double_night_bools = []
            for d in range(num_days - 2):
                # Night on d and night on d+2 means double night pair (夜明夜)
                b = model.NewBoolVar(f"dbl_night_n{n}_d{d}")
                model.AddBoolAnd([is_night[(n, d)], is_night[(n, d + 2)]]).OnlyEnforceIf(b)
                model.AddBoolOr([is_night[(n, d)].Not(), is_night[(n, d + 2)].Not()]).OnlyEnforceIf(b.Not())
                double_night_bools.append(b)
            model.Add(double_night_count == sum(double_night_bools))
            excess = model.NewIntVar(0, num_days, f"dbl_night_excess_n{n}")
            model.Add(excess >= double_night_count - max_double_night)
            model.Add(excess >= 0)
            penalties.append((excess, 500))

        # S6: Days off count target
        for n in N:
            off_count = model.NewIntVar(0, num_days, f"off_count_n{n}")
            # Off = OFF or AKE or KAN_AKE or NULL(0) ... actually count only explicit OFF
            # Count non-working days: off + ake + kan_ake
            all_off = []
            for d in D:
                not_work = model.NewBoolVar(f"not_work_n{n}_d{d}")
                model.AddBoolOr([is_off[(n, d)], is_ake[(n, d)], is_kan_ake[(n, d)]]).OnlyEnforceIf(not_work)
                model.AddBoolAnd([is_off[(n, d)].Not(), is_ake[(n, d)].Not(), is_kan_ake[(n, d)].Not()]).OnlyEnforceIf(not_work.Not())
                all_off.append(not_work)
            model.Add(off_count == sum(all_off))
            off_over = model.NewIntVar(0, num_days, f"off_over_n{n}")
            off_under = model.NewIntVar(0, num_days, f"off_under_n{n}")
            model.Add(off_count - max_days_off == off_over - off_under)
            penalties.append((off_over, 200))
            penalties.append((off_under, 200))

        # S7: Soft request constraints (日, 夜 etc.)
        soft_bonuses = []
        for n in N:
            nurse = active_nurses[n]
            nurse_id = str(nurse["id"])
            nurse_requests_map = requests.get(nurse_id, {})
            for day_str, req_type in nurse_requests_map.items():
                day_idx = int(day_str) - 1
                if 0 <= day_idx < num_days and req_type not in HARD_REQUEST and req_type in REQUEST_MAP:
                    target = REQUEST_MAP[req_type]
                    matched = model.NewBoolVar(f"soft_req_n{n}_d{day_idx}")
                    model.Add(shifts[(n, day_idx)] == target).OnlyEnforceIf(matched)
                    model.Add(shifts[(n, day_idx)] != target).OnlyEnforceIf(matched.Not())
                    soft_bonuses.append(matched)

        # --- Forbidden previous solutions ---
        for forbidden in forbidden_solutions:
            diffs = []
            for n in N:
                nurse_id = str(active_nurses[n]["id"])
                if nurse_id in forbidden:
                    for d in D:
                        prev_val = forbidden[nurse_id][d]
                        b = model.NewBoolVar(f"diff_pat{pat_idx}_n{n}_d{d}")
                        model.Add(shifts[(n, d)] != prev_val).OnlyEnforceIf(b)
                        model.Add(shifts[(n, d)] == prev_val).OnlyEnforceIf(b.Not())
                        diffs.append(b)
            if diffs:
                model.Add(sum(diffs) >= 1)

        # --- Objective ---
        total_penalty = model.NewIntVar(0, 10000000, "total_penalty")
        model.Add(total_penalty == sum(var * weight for var, weight in penalties))

        bonus_sum = model.NewIntVar(0, 10000000, "bonus_sum")
        if soft_bonuses:
            model.Add(bonus_sum == sum(b * 200 for b in soft_bonuses))
        else:
            model.Add(bonus_sum == 0)

        score_var = model.NewIntVar(-10000000, 10000000, "score")
        model.Add(score_var == 10000 - total_penalty + bonus_sum)

        model.Minimize(total_penalty - bonus_sum)

        # --- Solve ---
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 30
        solver.parameters.num_workers = 4

        status = solver.Solve(model)

        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            solution_data = {}
            solution_raw = {}
            for n in N:
                nurse_id = str(active_nurses[n]["id"])
                schedule = []
                raw = []
                for d in D:
                    val = solver.Value(shifts[(n, d)])
                    raw.append(val)
                    schedule.append(SHIFT_LABELS.get(val, ""))
                solution_data[nurse_id] = schedule
                solution_raw[nurse_id] = raw

            forbidden_solutions.append(solution_raw)

            # Calculate metrics
            night_counts_vals = []
            total_request_count = 0
            matched_request_count = 0
            total_day_shortage = 0
            total_night_shortage = 0
            consec_violations = 0
            total_off_days = 0

            for n in N:
                nurse_id = str(active_nurses[n]["id"])
                nc = sum(1 for d in D if solver.Value(shifts[(n, d)]) == NIGHT)
                night_counts_vals.append(nc)

                # Count off days
                off_d = sum(1 for d in D if solver.Value(shifts[(n, d)]) in (OFF, AKE, KAN_AKE))
                total_off_days += off_d

                # Request matching
                nurse_requests_map = requests.get(nurse_id, {})
                for day_str, req_type in nurse_requests_map.items():
                    day_idx = int(day_str) - 1
                    if 0 <= day_idx < num_days and req_type in REQUEST_MAP:
                        total_request_count += 1
                        if solver.Value(shifts[(n, day_idx)]) == REQUEST_MAP[req_type]:
                            matched_request_count += 1

                # Consecutive violations
                consec = 0
                for d in D:
                    if solver.Value(shifts[(n, d)]) in (DAY, NIGHT, KAN_NIGHT):
                        consec += 1
                        if consec > max_consec:
                            consec_violations += 1
                    else:
                        consec = 0

            for d in D:
                pattern_idx = d % len(night_pattern)
                required_night = night_pattern[pattern_idx]
                actual_night = sum(1 for n in N if solver.Value(shifts[(n, d)]) == NIGHT)
                if actual_night < required_night:
                    total_night_shortage += required_night - actual_night

                required_day = weekend_day_staff if d in weekends else weekday_day_staff
                actual_day = sum(1 for n in N if solver.Value(shifts[(n, d)]) == DAY)
                if actual_day < required_day:
                    total_day_shortage += required_day - actual_day

            night_balance = (max(night_counts_vals) - min(night_counts_vals)) if night_counts_vals else 0
            request_match = (matched_request_count / total_request_count * 100) if total_request_count > 0 else 100.0
            avg_days_off = total_off_days / num_nurses if num_nurses > 0 else 0

            score = solver.Value(score_var)

            results.append({
                "label": pattern_labels[pat_idx] if pat_idx < len(pattern_labels) else f"パターン{pat_idx + 1}",
                "data": solution_data,
                "score": score,
                "metrics": {
                    "nightBalance": round(night_balance, 1),
                    "dayShortage": total_day_shortage,
                    "nightShortage": total_night_shortage,
                    "consecViolations": consec_violations,
                    "requestMatch": round(request_match, 1),
                    "avgDaysOff": round(avg_days_off, 1),
                },
            })
        else:
            results.append({
                "label": pattern_labels[pat_idx] if pat_idx < len(pattern_labels) else f"パターン{pat_idx + 1}",
                "data": {},
                "score": 0,
                "metrics": {
                    "nightBalance": 0,
                    "dayShortage": 0,
                    "nightShortage": 0,
                    "consecViolations": 0,
                    "requestMatch": 0,
                    "avgDaysOff": 0,
                },
            })

    return results
