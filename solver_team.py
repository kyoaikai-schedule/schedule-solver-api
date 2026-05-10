"""夜勤チーム編成 ソルバ拡張 (Phase 2)

既存の solver.py には一切手を加えず、本ファイル単体で /solve_team の
ロジックを完結させる。基本的なモデル構築 (3シフト・夜→明→休 chain・
連続勤務・夜勤NG・人数要件 …) は solver._solve_one_pattern とほぼ同じ
構造を再実装し、その上にチーム関連のソフト制約を加算する。

3段フォールバック:
  relax_team=0  : チーム制約あり、ペナルティ強 (PENALTY_TEAM_MISSING=100)
  relax_team=1  : チーム制約あり、ペナルティ弱 (PENALTY_TEAM_MISSING=20)
  relax_team=2  : チーム制約完全に外す = 既存 solve_schedule をそのまま呼ぶ
"""

from __future__ import annotations
import math
import sys
import time
from ortools.sat.python import cp_model

# solver.py から読み取り専用で必要なものを import
# (solver.py は touch しない)
from solver import (
    DAY, NIGHT, OFF,
    SOLVER_VAL_TO_LABEL,  # 未使用だが互換性のため
    _build_night_req_table,
    _build_forced,
    _post_process,
    _validate,
    _preflight_diagnostics,
    solve_schedule,  # relax_team=2 のフォールバックで使う
)


def _log(msg: str) -> None:
    print(f"[solver_team] {msg}", file=sys.stderr, flush=True)


# ペナルティ重み (改善2: 100% 達成を目指して大幅増強)
# 他制約のソフトペナルティとの比較:
#   night balance diff: 300, 2連 night excess: 500, off_over/under: 200,
#   night_dev_penalties (relax2): 5000, day_short (relax1+): 5000
# チーム制約は night balance より優先、人数要件 (5000) は侵さないバランス
PENALTY_TEAM_MISSING_STRONG = 1000  # relax_team=0  (旧 100 → 1000)
PENALTY_TEAM_MISSING_WEAK = 200     # relax_team=1  (旧  20 →  200)
PENALTY_TEAM_OVERLAP = 300          # 重複 (旧 30 → 300)
PENALTY_TEAM_RESTING_WORK = 300     # 休み予定チームの夜勤 (旧 30 → 300)


def _used_teams_count(night_pattern: list[int]) -> int:
    """nightShiftPattern からチーム数を決定 (max を採用)"""
    if not night_pattern:
        return 0
    return max(int(p) for p in night_pattern)


def _team_letters(count: int) -> list[str]:
    """0..N から ['A','B',...] を生成 (最大5)"""
    return ['A', 'B', 'C', 'D', 'E'][:max(0, min(5, count))]


def _check_team_feasibility(
    active_nurses: list,
    night_req_table: list,
    used_teams: list[str],
    requests: dict,
    prev_month: dict,
    max_night_default: int,
) -> dict:
    """各チームから 1 名ずつ夜勤に配置することが数学的に可能か診断。

    各チーム T について:
      capacity_T = sum(maxNightShifts) for nurses in T (noNightShift 除外)
      demand_T = expected_T (夜勤人数 ≥ team-rank の日数)
    capacity_T < demand_T なら数学的に達成不可。
    """
    issues: list = []
    per_team_info: dict = {}

    # 各チームの demand: チーム T が expected に含まれる日数を計算
    # T は used_teams の 0-based index に対応 (A=0, B=1, ...)
    team_demand: dict = {t: 0 for t in used_teams}
    for d, req in enumerate(night_req_table):
        if req <= 0:
            continue
        # required = req のとき expected = used_teams[:req]
        for t in used_teams[:min(req, len(used_teams))]:
            team_demand[t] += 1

    # 各チームの capacity (夜勤可能ナースの maxNightShifts 合計)
    for team in used_teams:
        team_nurses = [n for n in active_nurses
                       if n.get("team") == team and not n.get("noNightShift", False)]
        capacity = sum(int(n.get("maxNightShifts", max_night_default)) for n in team_nurses)
        demand = team_demand.get(team, 0)
        per_team_info[team] = {
            "count": len(team_nurses),
            "capacity": capacity,
            "demand": demand,
        }
        if capacity < demand:
            issues.append({
                "team": team,
                "nurseCount": len(team_nurses),
                "capacity": capacity,
                "demand": demand,
                "shortage": demand - capacity,
                "reason": (
                    f"チーム{team} の夜勤可能容量 {capacity} (={len(team_nurses)}名 × maxNight) "
                    f"< 月内必要数 {demand}、不足 {demand - capacity}"
                ),
            })

    # 希望休/前月で当日チーム全員が forced 不在になる日も検出
    nid_to_team = {str(n["id"]): n.get("team") for n in active_nurses}
    nurses_in_team: dict = {t: set() for t in used_teams}
    for n in active_nurses:
        t = n.get("team")
        if t and t in nurses_in_team and not n.get("noNightShift", False):
            nurses_in_team[t].add(str(n["id"]))

    forced_off_per_day: dict = {}  # day_idx -> {nid: True}
    for nid, reqs in (requests or {}).items():
        if not isinstance(reqs, dict):
            continue
        for ds, lbl in reqs.items():
            try:
                d = int(ds) - 1
            except (TypeError, ValueError):
                continue
            if lbl in ("休", "有", "明", "管明"):
                forced_off_per_day.setdefault(d, set()).add(str(nid))
    for nid, prev in (prev_month or {}).items():
        if not isinstance(prev, dict):
            continue
        for k, lbl in prev.items():
            if str(k).startswith("_"):
                continue
            try:
                d = int(k) - 1
            except (TypeError, ValueError):
                continue
            if lbl in ("休", "有", "明", "管明"):
                forced_off_per_day.setdefault(d, set()).add(str(nid))

    blocked_days: list = []
    for d, req in enumerate(night_req_table):
        if req <= 0:
            continue
        expected = used_teams[:min(req, len(used_teams))]
        for team in expected:
            available = nurses_in_team.get(team, set()) - forced_off_per_day.get(d, set())
            if not available:
                blocked_days.append({
                    "day": d + 1,
                    "team": team,
                    "reason": f"day{d+1}: チーム{team} の夜勤可能ナース全員が forced 休/有/明 状態",
                })

    is_fully_feasible = len(issues) == 0 and len(blocked_days) == 0
    diagnosis = (
        "数学的に 100% 達成可能" if is_fully_feasible
        else f"達成不可能日が想定される (容量不足 {len(issues)} チーム、forced 不在 {len(blocked_days)} 日)"
    )

    # 達成可能上限率 (current max rate):
    # 各日について、その日 expected な全 team が capacity を持つかを確認
    # team_demand_actual / team_demand_max でざっくり計算
    total_days_with_req = sum(1 for r in night_req_table if r > 0)
    if total_days_with_req == 0:
        current_max_rate = 1.0
    else:
        # 不可能日: 容量不足チーム ⇒ shortage 日数 + blocked_days
        shortage_total = sum(i.get("shortage", 0) for i in issues)
        blocked_count = len({(b["day"], b["team"]) for b in blocked_days})
        unachievable_total = shortage_total + blocked_count
        achievable_days = max(0, total_days_with_req - unachievable_total)
        current_max_rate = round(achievable_days / total_days_with_req, 3)

    return {
        "isFullyFeasible": is_fully_feasible,
        "currentMaxRate": current_max_rate,
        "perTeamInfo": per_team_info,
        "issues": issues,
        "blockedDays": blocked_days[:30],
        "diagnosis": diagnosis,
    }


def _generate_improvement_suggestions(
    active_nurses: list,
    feasibility: dict,
    max_night_default: int = 7,
) -> list:
    """100% 達成不可能な場合の具体的な改善提案を生成。

    feasibility.perTeamInfo を見て不足チームを特定し、以下 3 つを提案:
      1. 「夜勤可能だが maxNight が低い」ナースの上限を引き上げる
      2. 「noNightShift=true」のナースを夜勤可能にする
      3. 「他チームから夜勤可能ナースを異動」(他チーム余裕ありのとき)

    返り値: priority 昇順, expectedCapacity 降順 で上位 5 件まで
    """
    suggestions: list = []
    team_stats = feasibility.get("perTeamInfo", {}) or {}

    for team, stats in team_stats.items():
        capacity = int(stats.get("capacity", 0))
        demand = int(stats.get("demand", 0))
        if demand <= 0 or capacity >= demand:
            continue  # このチームは問題なし

        shortage = demand - capacity
        team_nurses = [n for n in active_nurses if n.get("team") == team]
        active_team_nurses = [n for n in team_nurses if not n.get("noNightShift", False)]

        # 提案1: maxNightShifts を引き上げ
        if active_team_nurses:
            current_max_avg = (
                sum(int(n.get("maxNightShifts", max_night_default)) for n in active_team_nurses)
                / len(active_team_nurses)
            )
            # 必要な平均値: ceil(demand / 夜勤可能人数)
            required_avg = math.ceil(demand / len(active_team_nurses))
            if 0 < required_avg <= 12:  # 12回上限なら現実範囲とみなす
                # maxNight が低い順に shortage 名を抽出
                lowest = sorted(
                    active_team_nurses,
                    key=lambda n: int(n.get("maxNightShifts", max_night_default)),
                )[: max(1, shortage)]
                names = [n.get("name", f"id{n.get('id')}") for n in lowest]
                expected_capacity = len(active_team_nurses) * required_avg
                suggestions.append({
                    "priority": 1,
                    "team": team,
                    "type": "increase_max_night_shifts",
                    "title": f"チーム{team}のmaxNightShiftsを引き上げる",
                    "description": (
                        f"チーム{team}の夜勤可能{len(active_team_nurses)}名の月内夜勤上限"
                        f"(現状平均{current_max_avg:.1f}回)を {required_avg}回以上に引き上げる。"
                        f"特に上限が低い{len(names)}名 ({', '.join(names)}) を見直すと効果的"
                    ),
                    "targetNurses": names,
                    "currentCapacity": capacity,
                    "expectedCapacity": expected_capacity,
                    "expectedDemand": demand,
                    "shortage": shortage,
                    "feasibility": "easy",
                })

        # 提案2: noNightShift=true ナースを夜勤可能にする
        no_night_nurses = [n for n in team_nurses if n.get("noNightShift", False)]
        for nurse in no_night_nurses:
            added = int(nurse.get("maxNightShifts", max_night_default)) or max_night_default
            if added <= 0:
                added = max_night_default
            new_capacity = capacity + added
            nurse_name = nurse.get("name") or f"id{nurse.get('id')}"
            position = nurse.get("position", "")
            achievement_note = (
                "達成" if new_capacity >= demand
                else f"満たすには更に{demand - new_capacity}不足"
            )
            suggestions.append({
                "priority": 2,
                "team": team,
                "type": "enable_night_shift",
                "title": f"{nurse_name}を夜勤可能にする",
                "description": (
                    f"チーム{team}の{nurse_name}"
                    f"({position}) は現在 noNightShift=true (夜勤不可)。"
                    f"夜勤可能にすると容量が {capacity}→{new_capacity}に増加 "
                    f"(必要 {demand} を{achievement_note})"
                ),
                "targetNurses": [nurse_name],
                "currentCapacity": capacity,
                "expectedCapacity": new_capacity,
                "expectedDemand": demand,
                "shortage": shortage,
                "feasibility": "medium",
            })

        # 提案3: 他チームからの異動 (他チームが余裕あり)
        for other_team, other_stats in team_stats.items():
            if other_team == team:
                continue
            other_capacity = int(other_stats.get("capacity", 0))
            other_demand = int(other_stats.get("demand", 0))
            other_surplus = other_capacity - other_demand
            if other_surplus <= 0:
                continue
            other_nurses_active = [
                n for n in active_nurses
                if n.get("team") == other_team and not n.get("noNightShift", False)
            ]
            if len(other_nurses_active) <= 1:
                continue  # 1 名のみは異動候補外
            # maxNight が高いナースを移籍候補
            candidate = max(
                other_nurses_active,
                key=lambda n: int(n.get("maxNightShifts", max_night_default)),
            )
            cand_max = int(candidate.get("maxNightShifts", max_night_default))
            new_to_capacity = capacity + cand_max
            new_from_capacity = other_capacity - cand_max
            # 異動後に from チームの容量が demand を割らない場合のみ提案
            if new_from_capacity < other_demand:
                continue
            suggestions.append({
                "priority": 3,
                "team": team,
                "type": "transfer_nurse",
                "title": f"{candidate.get('name')}をチーム{other_team}→{team}に異動",
                "description": (
                    f"チーム{other_team}は容量{other_capacity}/必要{other_demand}で"
                    f"{other_surplus}余裕あり。{candidate.get('name')}"
                    f"(maxNight={cand_max})をチーム{team}に異動すると"
                    f"チーム{team} 容量 {capacity}→{new_to_capacity} に改善"
                    f"(チーム{other_team} は {other_capacity}→{new_from_capacity} で必要を満たす)"
                ),
                "targetNurses": [candidate.get("name", f"id{candidate.get('id')}")],
                "fromTeam": other_team,
                "toTeam": team,
                "currentCapacity": capacity,
                "expectedCapacity": new_to_capacity,
                "expectedDemand": demand,
                "shortage": shortage,
                "feasibility": "medium",
            })

    # priority 昇順、expectedCapacity 降順 (より大きく改善するものを優先)
    suggestions.sort(key=lambda s: (s.get("priority", 99), -s.get("expectedCapacity", 0)))
    return suggestions[:5]


def _team_diagnostics(active_nurses: list, night_pattern: list[int]) -> dict:
    """フェーズ2 のチーム関連 preflight 診断"""
    used_count = _used_teams_count(night_pattern)
    required_teams = _team_letters(used_count)
    per_team_count: dict = {t: 0 for t in required_teams}
    nurses_with_team = 0
    nurses_without_team = 0

    for n in active_nurses:
        team = n.get("team")
        if team and isinstance(team, str) and team in per_team_count:
            per_team_count[team] += 1
            nurses_with_team += 1
        elif team and isinstance(team, str):
            # 'F' 等、想定外のチーム文字
            per_team_count[team] = per_team_count.get(team, 0) + 1
            nurses_with_team += 1
        else:
            nurses_without_team += 1

    warnings = []
    if not required_teams:
        warnings.append("nightShiftPattern が空のためチームモードを無効化します")
    else:
        # 各チーム最低1人は必要 (夜勤を担当できるナース)
        for t in required_teams:
            if per_team_count.get(t, 0) == 0:
                warnings.append(f"チーム{t} に所属するナースが 0 名 (夜勤を割り当て不能)")
        avg = (sum(per_team_count.values()) /
               len(required_teams)) if required_teams else 0
        for t in required_teams:
            cnt = per_team_count.get(t, 0)
            if avg > 0 and cnt < avg * 0.6:
                warnings.append(f"チーム{t} が {cnt}名 (他チーム比較で少ない)")
    if nurses_without_team > 0:
        warnings.append(f"{nurses_without_team}名がチーム未設定です (チーム制約に縛られず配置)")

    return {
        "teamMode": True,
        "requiredTeams": required_teams,
        "nursesWithTeam": nurses_with_team,
        "nursesWithoutTeam": nurses_without_team,
        "perTeamCount": per_team_count,
        "warnings": warnings,
    }


def _solve_one_pattern_with_teams(
    params: dict,
    forbidden_solutions: list,
    relax_team: int,
    relax_level: int = 0,
    pat_idx: int = 0,
) -> dict:
    """solver._solve_one_pattern と同じ制約 + チームペナルティ。

    relax_team:
      0 → チームペナルティ強 (PENALTY_TEAM_MISSING_STRONG)
      1 → チームペナルティ弱 (PENALTY_TEAM_MISSING_WEAK)
    pat_idx:
      0..N-1, 各パターンに異なる random_seed を割り当てて多様性を担保
    relax_level:
      日勤・夜勤人数要件の緩和度合い (既存と同じ意味、本関数内では 0 固定で OK)
    """
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
    used_teams = params["used_teams"]            # ['A','B','C','D'] 等
    team_of_nurse_idx = params["team_of_nurse_idx"]  # n_idx -> 'A'|'B'|...|None

    num_nurses = len(active_nurses)
    N = range(num_nurses)
    D = range(num_days)

    model = cp_model.CpModel()

    shifts: dict = {}
    is_day: dict = {}
    is_night: dict = {}
    is_off: dict = {}
    is_working: dict = {}

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

    # ハード制約: 夜→翌日OFF / 夜→翌々日OFF
    for n in N:
        for d in D:
            if d + 1 < num_days:
                model.add(is_off[(n, d + 1)] == 1).only_enforce_if(is_night[(n, d)])
            if d + 2 < num_days:
                model.add(is_off[(n, d + 2)] == 1).only_enforce_if(is_night[(n, d)])

    # forced 休/有 の前日 NIGHT 禁止
    for (n, d), lbl in forced_label.items():
        if lbl in ("休", "有") and d > 0 and (n, d - 1) not in forced_shift:
            model.add(is_night[(n, d - 1)] == 0)

    # 3連夜勤禁止
    for n in N:
        for d in range(num_days - 4):
            model.add_bool_or([
                is_night[(n, d)].negated(),
                is_night[(n, d + 2)].negated(),
                is_night[(n, d + 4)].negated(),
            ])

    # 連続勤務制限
    for n in N:
        nurse = active_nurses[n]
        nid = str(nurse["id"])
        prev_consec = prev_month.get(nid, {}).get("_consecDays", 0) \
            if isinstance(prev_month.get(nid), dict) else 0

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

    # 夜勤NGペア
    for pair in night_ng_pairs:
        if len(pair) < 2:
            continue
        idx_a = next((i for i, nn in enumerate(active_nurses) if nn["id"] == pair[0]), None)
        idx_b = next((i for i, nn in enumerate(active_nurses) if nn["id"] == pair[1]), None)
        if idx_a is not None and idx_b is not None:
            for d in D:
                model.add_bool_or([is_night[(idx_a, d)].negated(), is_night[(idx_b, d)].negated()])

    # 個人設定
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

    # 夜勤人数 (relax_level<2 で完全一致、>=2 で ±1)
    night_dev_penalties: list = []
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

    # 日勤人数 (relax_level==0 で完全、>=1 で -1 + ペナルティ)
    day_short_penalties: list = []
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

    # ===========================
    # ソフト制約 (ペナルティ)
    # ===========================
    penalties: list = []
    penalties.extend(night_dev_penalties)
    penalties.extend(day_short_penalties)

    # 夜勤回数の均等化
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

    # 休日数 (休+有 のみ)
    for n in N:
        ake_kanake_terms = []
        for d in D:
            forced = forced_label.get((n, d))
            if forced in ("明", "管明"):
                ake_kanake_terms.append(1)
            elif forced is not None:
                continue
            else:
                if d > 0:
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

    # ===========================
    # 既出解の禁止 (改善2 パターン戦略)
    #   pat_idx=0 → forbidden 制約なし (純粋に最適 = balanceRate 最大)
    #   pat_idx=1 → 25% 差以上 (pat0 から軽微変更)
    #   pat_idx=2 → 50% 差以上 (pat0 から大きく変更)
    # ===========================
    if pat_idx >= 1 and forbidden_solutions:
        # 直前のパターンとの差分のみ考慮 (累積禁止だと pat3 が過剰制約になる)
        diff_threshold = num_nurses // 4 if pat_idx == 1 else num_nurses // 2
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
                model.add(sum(diffs) >= max(1, diff_threshold))

    # ===========================
    # ★ チームペナルティ (本拡張のメイン)
    # ===========================
    missing_w = PENALTY_TEAM_MISSING_STRONG if relax_team == 0 else PENALTY_TEAM_MISSING_WEAK

    for d in D:
        required = night_req_table[d]
        if required <= 0:
            continue
        # 当日 expected な (= 夜勤に出るべき) チームと、休む (= rest) チーム
        expected_teams = used_teams[:required] if required <= len(used_teams) else used_teams
        resting_teams = [t for t in used_teams if t not in expected_teams]

        # チームごとの「この日の夜勤者数」を IntVar として表現
        for team in used_teams:
            team_idx_list = [n for n in N
                             if team_of_nurse_idx.get(n) == team
                             and forced_label.get((n, d)) != "管夜"]
            if not team_idx_list:
                # チーム所属者0人 → expected の場合は missing 1 として確定
                if team in expected_teams:
                    # 定数 missing
                    fake = model.new_int_var(1, 1, f"fakeMiss_{team}_{d}")
                    penalties.append((fake, missing_w))
                continue

            team_count = model.new_int_var(0, len(team_idx_list), f"tc_{team}_{d}")
            model.add(team_count == sum(is_night[(n, d)] for n in team_idx_list))

            if team in expected_teams:
                # missing: team_count == 0 → penalty
                missing = model.new_bool_var(f"miss_{team}_{d}")
                # team_count == 0  ⇔  missing
                model.add(team_count == 0).only_enforce_if(missing)
                model.add(team_count >= 1).only_enforce_if(missing.negated())
                penalties.append((missing, missing_w))

                # overlap: team_count >= 2 ならその超過量 (team_count - 1) を penalty
                overlap = model.new_int_var(0, len(team_idx_list), f"over_{team}_{d}")
                model.add(overlap >= team_count - 1)
                model.add(overlap >= 0)
                penalties.append((overlap, PENALTY_TEAM_OVERLAP))
            else:
                # resting team: 当日は夜勤に出ないのが理想 → 1人以上いたらペナルティ
                penalties.append((team_count, PENALTY_TEAM_RESTING_WORK))

    # ===========================
    # 目的関数
    # ===========================
    if penalties:
        total_penalty = sum(var * w for var, w in penalties)
        model.minimize(total_penalty)

    solver = cp_model.CpSolver()
    # 改善3: パターンごとに異なる random_seed を割り当てて多様性を担保
    solver.parameters.random_seed = 1000 + pat_idx * 137 + relax_team * 7
    # 改善2: 100% 達成を目指して探索時間を 15s → 60s に延長
    # (3パターン × 1 relax (relax=0で多くは収まる) ≈ 180s, frontend 300s timeout 内)
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_workers = 8
    solver.parameters.randomize_search = True
    # 改善2: 線形化レベル up + presolve で探索精度向上
    solver.parameters.linearization_level = 2
    solver.parameters.cp_model_presolve = True
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


def _compute_team_metrics(
    data: dict,
    active_nurses: list,
    night_req_table: list[int],
    used_teams: list[str],
    feasibility: dict | None = None,
) -> dict:
    """生成後の解からチームバランス指標を算出 + 達成不可能日を分析"""
    nid_to_team = {str(n["id"]): n.get("team") for n in active_nurses}

    per_day_balance = []
    balanced_count = 0
    total_days_with_req = 0
    unachievable: list = []
    blocked_set = set()
    if feasibility:
        for bd in feasibility.get("blockedDays", []):
            blocked_set.add((bd["day"], bd["team"]))

    for d in range(len(night_req_table)):
        required = night_req_table[d]
        if required <= 0:
            per_day_balance.append({
                "day": d + 1,
                "expected": [],
                "actual": [],
                "isBalanced": True,
                "missing": [],
                "extra": [],
            })
            continue

        total_days_with_req += 1
        expected = used_teams[:required] if required <= len(used_teams) else used_teams[:]

        # 当日夜勤者の team を集計
        actual_teams: list[str] = []
        for nid, shifts in data.items():
            if d < len(shifts) and shifts[d] == "夜":
                team = nid_to_team.get(nid)
                actual_teams.append(team if team else "未")

        # team 未設定 ('未') はメトリクス上は「中立」扱い。
        # 仕様: チーム未設定者は制約に縛られず、配置されてもペナルティに影響しない
        actual_named = [t for t in actual_teams if t != "未"]

        # missing = expected のうち actual_named に1人もいない
        actual_set = set(actual_named)
        missing = [t for t in expected if t not in actual_set]

        # extra =
        #   - expected で 2人以上いるチーム (overlap)
        #   - expected 以外の team (resting team が夜勤に出ているケース)
        #   未組はカウント対象外
        from collections import Counter
        cnt = Counter(actual_named)
        extra = []
        for t, c in cnt.items():
            if t in expected and c >= 2:
                extra.append(t)
            elif t not in expected:
                extra.append(t)

        is_balanced = len(missing) == 0 and len(extra) == 0
        if is_balanced:
            balanced_count += 1
        else:
            # 達成不可能日のレポート: missing にあるチームと、その理由を推測
            for m_team in missing:
                key = (d + 1, m_team)
                if key in blocked_set:
                    reason = f"チーム{m_team} の夜勤可能ナース全員が forced 休/有/明"
                else:
                    # 容量不足が原因か、ソルバーが他制約優先したか
                    cap_issue = next(
                        (i for i in (feasibility or {}).get("issues", [])
                         if i.get("team") == m_team), None) if feasibility else None
                    if cap_issue:
                        reason = f"チーム{m_team} の月内夜勤容量不足 (capacity={cap_issue['capacity']} < demand={cap_issue['demand']})"
                    else:
                        reason = f"チーム{m_team} 不在 (他制約のため solver が他の team を優先)"
                unachievable.append({
                    "day": d + 1,
                    "teamShortage": m_team,
                    "reason": reason,
                })

        per_day_balance.append({
            "day": d + 1,
            "expected": expected,
            "actual": sorted(actual_teams),
            "isBalanced": is_balanced,
            "missing": missing,
            "extra": sorted(extra),
        })

    balance_rate = (balanced_count / total_days_with_req
                    if total_days_with_req > 0 else 1.0)

    return {
        "perDayTeamBalance": per_day_balance,
        "balanceRate": round(balance_rate, 3),
        "balancedDays": balanced_count,
        "totalDays": total_days_with_req,
        "unachievableDays": unachievable[:50],  # 多すぎ防止
    }


def solve_with_teams(request_data: dict) -> dict:
    """新エンドポイント /solve_team の本体。
    既存 /solve と同じ JSON 形式を受けて、teamMetrics を含む結果を返す。
    """
    config = request_data.get("config", {})
    night_pattern = config.get("nightShiftPattern", [2, 2])
    used_team_count = _used_teams_count(night_pattern)
    used_teams = _team_letters(used_team_count)

    nurses = request_data.get("nurses", [])
    active_nurses = [n for n in nurses if not n.get("excludeFromGeneration", False)]
    team_diag = _team_diagnostics(active_nurses, night_pattern)

    _log(f"=== /solve_team start: pattern={night_pattern} usedTeams={used_teams} "
         f"nursesWithTeam={team_diag['nursesWithTeam']} "
         f"nursesWithoutTeam={team_diag['nursesWithoutTeam']}")
    for w in team_diag["warnings"]:
        _log(f"  TEAM PREFLIGHT: {w}")

    # チーム未定義 (used_teams が空) → 既存 solve に直行
    if not used_teams:
        _log("usedTeams 空のためチームモードを無効化、既存 solve_schedule にフォールバック")
        result = solve_schedule(request_data)
        # 各 pattern の metrics に teamMetrics を追加
        for p in result:
            m = p.get("metrics") or {}
            m["teamMetrics"] = {
                "teamMode": False,
                "teamCount": 0,
                "usedTeams": [],
                "fallbackLevel": 2,
                "attemptsTeam": [],
                "perDayTeamBalance": [],
                "balanceRate": None,
                "balancedDays": 0,
                "totalDays": 0,
                "diagnostics": team_diag,
            }
            p["metrics"] = m
        return {"patterns": result}

    # チーム別 nurse index マップ
    team_of_nurse_idx: dict = {}
    for i, n in enumerate(active_nurses):
        t = n.get("team")
        team_of_nurse_idx[i] = t if (isinstance(t, str) and t) else None

    # 共通パラメータ準備 (solver._build_forced 等を活用)
    num_days = request_data["daysInMonth"]
    year = request_data.get("year", 2026)
    month = request_data.get("month", 0)
    requests = request_data.get("requests", {})
    night_ng_pairs = request_data.get("nightNgPairs", [])
    prev_month = request_data.get("prevMonthConstraints", {})
    holidays = set(request_data.get("holidays", []))
    weekends = set(request_data.get("weekends", []))
    num_patterns = request_data.get("numPatterns", 3)

    weekday_day_staff = config["weekdayDayStaff"]
    weekend_day_staff = config["weekendDayStaff"]
    max_night = config.get("maxNightShifts", 6)
    max_days_off = config.get("maxDaysOff", 10)
    max_consec = config.get("maxConsecutiveDays", 3)
    max_double_night = config.get("maxDoubleNightPairs", 2)

    night_req_table = _build_night_req_table(
        year, month, num_days, night_pattern,
        start_with_three=config.get("startWithThree", False),
    )
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
        "used_teams": used_teams,
        "team_of_nurse_idx": team_of_nurse_idx,
    }

    # 改善2: 数学的達成可能性チェック (各チームが 100% 達成可能か事前判定)
    feasibility = _check_team_feasibility(
        active_nurses, night_req_table, used_teams,
        requests, prev_month, max_night,
    )
    _log(f"feasibility: isFullyFeasible={feasibility['isFullyFeasible']} "
         f"issues={len(feasibility['issues'])} blocked={len(feasibility['blockedDays'])} "
         f"currentMaxRate={feasibility['currentMaxRate']}")

    # 改善3: 100% 不可能なら改善提案を生成
    improvement_suggestions = []
    if not feasibility["isFullyFeasible"]:
        improvement_suggestions = _generate_improvement_suggestions(
            active_nurses, feasibility, max_night,
        )
        _log(f"improvement suggestions: {len(improvement_suggestions)} 件")

    # 既存 preflight も付加 (既存 solve と同条件で診断)
    preflight = _preflight_diagnostics(
        active_nurses, night_req_table, weekday_day_staff, weekend_day_staff,
        weekends_combined, num_days, max_night,
        max_consec=max_consec, max_days_off=max_days_off,
        prev_month=prev_month, requests=requests,
    )

    pattern_labels = ["パターンA", "パターンB", "パターンC", "パターンD", "パターンE"]
    forbidden_solutions: list = []
    results: list = []

    for pat_idx in range(num_patterns):
        label = pattern_labels[pat_idx] if pat_idx < len(pattern_labels) else f"パターン{pat_idx + 1}"
        _log(f"=== {label} 開始 (pat_idx={pat_idx}) ===")

        attempts_team: list = []
        chosen = None
        chosen_relax_team = None

        # 3段フォールバック
        for relax_team in (0, 1, 2):
            t0 = time.time()
            if relax_team == 2:
                # 既存 solve_schedule にフォールバック (1 パターン分のみ)
                _log(f"  {label} relax_team=2: 既存 solve_schedule にフォールバック")
                fallback_input = dict(request_data)
                fallback_input["numPatterns"] = 1
                fb_result = solve_schedule(fallback_input)
                elapsed = time.time() - t0
                if fb_result and fb_result[0].get("data"):
                    raw = {}
                    for nid, labels in fb_result[0]["data"].items():
                        # solve_schedule は labels (出力) を返す。raw はないので、
                        # 元の data をそのまま採用 (post-proc は既にされている)
                        raw[nid] = labels
                    attempts_team.append({
                        "relaxTeam": 2,
                        "status": "FALLBACK_EXISTING",
                        "elapsedSec": round(elapsed, 2),
                    })
                    # data は既に label 済み
                    chosen = {
                        "data_labels": fb_result[0]["data"],
                        "score": fb_result[0].get("score", 0),
                        "metrics_existing": fb_result[0].get("metrics", {}),
                    }
                    chosen_relax_team = 2
                    _log(f"  {label} relax_team=2: 採用 ({elapsed:.2f}s)")
                else:
                    attempts_team.append({
                        "relaxTeam": 2,
                        "status": "FALLBACK_FAILED",
                        "elapsedSec": round(elapsed, 2),
                    })
                break

            # relax_team 0 or 1: チームペナルティ付きで solve
            res = _solve_one_pattern_with_teams(
                params, forbidden_solutions, relax_team=relax_team, relax_level=0,
                pat_idx=pat_idx,
            )
            elapsed = time.time() - t0
            attempts_team.append({
                "relaxTeam": relax_team,
                "status": res["status"],
                "elapsedSec": round(elapsed, 2),
            })
            _log(f"  {label} relax_team={relax_team}: status={res['status']} elapsed={elapsed:.2f}s")

            if res["raw"] is None:
                continue

            data_labels = _post_process(res["raw"], active_nurses, forced_label, num_days)
            errors = _validate(data_labels, params, relax_level=0)
            if not errors:
                chosen = {
                    "raw": res["raw"],
                    "data_labels": data_labels,
                    "objective": res["objective"],
                }
                chosen_relax_team = relax_team
                _log(f"  {label} relax_team={relax_team}: 採用")
                break
            _log(f"  {label} relax_team={relax_team}: validation NG ({len(errors)}件) → 次へ")

        if chosen is None:
            # 全レベル失敗
            results.append({
                "label": label,
                "data": {},
                "score": 0,
                "metrics": {
                    "solverUsed": True,
                    "error": "解が見つかりませんでした (チームモード全レベル失敗)",
                    "relaxLevel": -1,
                    "nightBalance": 0, "dayShortage": 0, "nightShortage": 0,
                    "consecViolations": 0, "requestMatch": 0, "avgDaysOff": 0, "nullCells": 0,
                    "teamMetrics": {
                        "teamMode": True,
                        "teamCount": len(used_teams),
                        "usedTeams": used_teams,
                        "fallbackLevel": -1,
                        "attemptsTeam": attempts_team,
                        "perDayTeamBalance": [],
                        "balanceRate": None,
                        "balancedDays": 0,
                        "totalDays": 0,
                        "diagnostics": team_diag,
                    },
                    "diagnostics": preflight,
                },
            })
            continue

        # チームメトリクス計算
        if "raw" in chosen:
            forbidden_solutions.append(chosen["raw"])
        team_metrics = _compute_team_metrics(
            chosen["data_labels"], active_nurses, night_req_table, used_teams,
            feasibility=feasibility,
        )
        team_metrics.update({
            "teamMode": True,
            "teamCount": len(used_teams),
            "usedTeams": used_teams,
            "fallbackLevel": chosen_relax_team,
            "attemptsTeam": attempts_team,
            "diagnostics": team_diag,
            "feasibility": feasibility,
            "improvementSuggestions": improvement_suggestions,
        })

        # 既存類似のメトリクス計算 (簡略)
        night_vals = [
            sum(1 for s in chosen["data_labels"].get(str(n["id"]), []) if s == "夜")
            for n in active_nurses
        ]
        night_balance = (max(night_vals) - min(night_vals)) if night_vals else 0
        total_off = sum(sum(1 for s in sh if s in ("休", "有"))
                        for sh in chosen["data_labels"].values())
        avg_off = total_off / len(active_nurses) if active_nurses else 0

        score = chosen.get("score") or max(0, 10000 - chosen.get("objective", 0))

        results.append({
            "label": label,
            "data": chosen["data_labels"],
            "score": int(score),
            "metrics": {
                "solverUsed": True,
                "relaxLevel": 0,
                "nightBalance": round(night_balance, 1),
                "avgDaysOff": round(avg_off, 1),
                "teamMetrics": team_metrics,
                "diagnostics": preflight,
            },
        })

    return {"patterns": results}
