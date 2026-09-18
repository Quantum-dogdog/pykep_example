import itertools
import math
import time
import sys
import os
import zlib
import pygmo as pg
from pykep import epoch
from pykep.planet import jpl_lp
from pykep.trajopt import mga_1dsm
from multiprocessing import Pool, cpu_count

# ============================== 全局参数 ==============================
SADE_GEN = 1500                 # SADE 每次进化代数(全局搜索主力)
SCAN_SEEDS = [1, 2, 3]          # 扫描阶段每序列独立重启种子(取最优)
RESTART_SEEDS = [1, 2, 3, 4, 5] # 精算阶段 Top-N 序列使用的种子
BUDGET_SEEDS = [21, 22, 23, 24, 25, 26]   # 预算追加轮种子(避开已用过的)
KEY_FEED_SEEDS = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40]  # 重点序列加喂种子
TOP_N_REFINE = 10               # 进入精算的前排序列数
COMPASS_FEVALS = 5000           # COMPASS 局部精修最大评估次数
BUDGET_GAP_KMS = 0.3            # SADE→COMPASS 差距超过此值判为欠收敛
BUDGET_ROUNDS = 2               # 预算追加轮数上限(防死循环)

RESONANCE_HARMONICS = (1, 1.5, 2, 2.5, 3, 4)  # 谐波含半整数(3:2曾被自发找到)
RESONANCE_BASE_TOL = 15.0       # 共振窗口基础容差(天)
RESONANCE_MODE = 'off'          # 'off'=共振段自由(公平基准) | 'hard' | 'soft'
RESONANCE_SOFT_WEIGHT = 30.0    # soft 模式软罚权重(m/s/天)
FAIR_CAP_RATIO = 2.0            # 公平时长帽: 每序列 = 自身TOF下界和 × 此倍数
FAIR_CAP_MAX_YEARS = 9.5        # 时长帽绝对封顶(年)
FEASIBILITY_PENALTY = 1e9       # 不可行解罚值(m/s)
INFEASIBLE_THRESHOLD = 1e8      # 判定"未找到可行解"的阈值
WINDOW_EPS = 1e-6               # 共振窗口边缘收缩量
TIME_WEIGHT = 0.0               # 可选软时间罚(默认关)
T0_START_MJD2000 = 10227.0      # 发射窗口起点 2028-01-01 (天问四备选窗)
T0_END_MJD2000 = 11323.0        # 发射窗口终点 2031-01-01

# —— ΔV 计费口径 ——
ARRIVAL_MODE = 'capture'        # 'capture'=到达按JOI捕获烧计费 | 'full'=旧口径
CHARGE_DEPARTURE = True         # 出发v∞计入总ΔV, 报表单列"火箭承担"
DEPARTURE_VINF_MAX = 5.6        # 出发超速上限 km/s ≈ C3 31.4 (朱诺实况)
JOI_RP_RADII = 6.0              # JOI 近木点(木星半径倍数, 6R_J 避开辐射带)
JOI_PERIOD_DAYS = 300.0         # JOI 捕获目标轨道周期(天)
MU_J = 1.26686534e8             # 木星引力常数 km^3/s^2
R_J = 71492.0                   # 木星半径 km

# —— 重点加喂序列(史实架构对照: Juno 型 / Galileo 型变体) ——
KEY_FEED = [('earth', 'earth', 'jupiter'),                  # E-E-J, 航天器推进最低
            ('earth', 'earth', 'venus', 'earth', 'earth', 'jupiter')]  # E-E-V-E-E-J

PERIODS = {'earth': 365.25, 'venus': 224.7, 'mars': 687.0}  # 公转周期(天)
DV_TO_KMS = 1e-3               # fitness(m/s) → km/s


class LayoutError(Exception):
    """udp 决策变量布局解析失败时抛出"""
    pass


# ====================== 动态边界: 全部按序列结构生成 ======================
def pop_size_for(dim):
    """种群按维度自适应: 2维50, 22维176"""
    return max(50, min(200, 8 * dim))


def generate_sequences():
    """枚举 E→(0~4个中间行星)→J 全部 121 种序列"""
    planets = ['earth', 'venus', 'mars']
    all_seqs = []
    for r in range(0, 5):                                     # 中间行星个数 0~4
        for combo in itertools.product(planets, repeat=r):    # 有序笛卡尔积
            all_seqs.append(['earth'] + list(combo) + ['jupiter'])
    return all_seqs


def seq_display(seq_names):
    """E -> V -> J 紧凑显示"""
    return " -> ".join(s[0].upper() for s in seq_names)


def seed_from_seq(seq_names):
    """序列名→确定性种子(备用工具)"""
    return zlib.crc32("->".join(seq_names).encode()) % (2 ** 31)


def get_tof_bounds(p1, p2, is_final=False):
    """逐航段 TOF 包络(天): 按行星对查询, 末段单独放宽"""
    Y = 365.25
    if is_final:                                              # 末段(抵木星前)
        return [2.5 * Y, 7.0 * Y]                             # 2.5~7 年慢转移
    pair = {p1, p2}                                           # 无序对, 正反向共用
    if pair == {'earth', 'venus'}: return [100, 250]
    elif pair == {'earth', 'mars'}: return [200, 500]
    elif pair == {'earth', 'earth'}: return [340, 800]
    elif pair == {'venus', 'venus'}: return [290, 460]
    elif pair == {'mars', 'mars'}: return [680, 1400]
    elif pair == {'venus', 'mars'}: return [200, 600]
    raise ValueError(f"未知的行星对: {p1} -> {p2}")


def resonance_tol(k, period):
    """共振容差随 k 放宽(高阶周期长, 容差按比例放大)"""
    return max(RESONANCE_BASE_TOL, 0.03 * k * period)


def get_resonant_tof_bounds(planet_name, default_bounds):
    """hard 模式专用: 共振段收窄到最宽谐波窗口"""
    period = PERIODS[planet_name]
    best = None
    for k in RESONANCE_HARMONICS:
        tol = resonance_tol(k, period)                        # 该谐波容差
        lo = max(k * period - tol + WINDOW_EPS, default_bounds[0])
        hi = min(k * period + tol - WINDOW_EPS, default_bounds[1])
        if hi > lo and (best is None or (hi - lo) > (best[1] - best[0])):
            best = [lo, hi]                                   # 保留最宽窗口
    return best


def get_vinf_bounds(seq_names):
    """发射 v∞ 包络: 按借力次数给, 统一按火箭能力封顶"""
    n_assist = len(seq_names) - 2
    if n_assist == 0: b = [6.0, 12.0]                         # 直飞本能要求高能
    elif n_assist == 1: b = [2.5, 9.0]
    elif n_assist == 2: b = [2.5, 8.0]
    else: b = [2.5, 6.0]
    b[1] = min(b[1], DEPARTURE_VINF_MAX)                      # 火箭能力上限(C3≈31)
    b[0] = min(b[0], b[1] - 0.5)                              # 保证 lo<hi 有效
    return b


def get_flyby_bounds(planet_name):
    """飞掠包络: beta 全周, rp 下界按行星大气/障碍高度"""
    p = math.pi
    if planet_name == 'earth': rp_bounds = [1.045, 5.0]       # 避开大气层
    elif planet_name == 'venus': rp_bounds = [1.1, 5.0]       # 避开浓密大气
    elif planet_name == 'mars': rp_bounds = [1.1, 4.0]        # 避开稀薄大气
    else: rp_bounds = [1.0, 10.0]                             # 兜底
    return [-p, p], rp_bounds


def get_time_cap_days(tof_bounds):
    """公平总时长帽: min(自身下界和×倍数, 绝对封顶)"""
    lb_sum = sum(b[0] for b in tof_bounds)                    # 该序列最紧时间线
    return min(FAIR_CAP_RATIO * lb_sum, FAIR_CAP_MAX_YEARS * 365.25)


def joi_capture_dv_kms(vinf_kms, rp_radii=None, period_days=None):
    """到达超速 → 近木点 JOI 捕获烧(km/s), Oberth 效应使捕获便宜.
    校验: v∞=5.6, rp=1.06R_J, 捕获53.5天 → 0.539 (朱诺实测542 m/s) ✓"""
    rp_radii = JOI_RP_RADII if rp_radii is None else rp_radii
    period_days = JOI_PERIOD_DAYS if period_days is None else period_days
    rp = rp_radii * R_J                                       # 近木点(km)
    a = (MU_J * (period_days * 86400.0 / (2.0 * math.pi)) ** 2) ** (1.0/3.0)
    v_arr = math.sqrt(vinf_kms ** 2 + 2.0 * MU_J / rp)        # 到达双曲线近木点速
    v_cap = math.sqrt(MU_J * (2.0 / rp - 1.0 / a))            # 捕获椭圆近木点速
    return max(v_arr - v_cap, 0.0)                            # 一次点火减速量


# ==================== udp 决策变量布局解析(自动适配) ====================
def resolve_layout(raw_udp, t0_bounds, tof_bounds, vinf_bounds,
                   add_vinf_dep, add_vinf_arr):
    """按边界值反查 mga_1dsm 内部变量槽位(t0/tof/vinf/beta/rp/alpha)"""
    lb, ub = raw_udp.get_bounds()
    lb, ub = list(lb), list(ub)
    n = len(lb)
    n_legs = len(tof_bounds)

    def eq(a, b):                                             # 浮点边界相等判断
        return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))

    t0_lo, t0_hi = t0_bounds[0].mjd2000, t0_bounds[1].mjd2000
    two_pi = 2.0 * math.pi
    vinf_mps = (vinf_bounds[0] * 1000.0, vinf_bounds[1] * 1000.0)

    t0_idx = vinf_idx = -1
    tof_slots, beta_slots, rp_slots = [], [], []
    n_alpha = 0
    next_leg = 0

    for i in range(n):                                        # 逐变量按边界特征认领
        lo, hi = lb[i], ub[i]
        if t0_idx < 0 and eq(lo, t0_lo) and eq(hi, t0_hi):
            t0_idx = i; continue                              # ← t0
        if next_leg < n_legs and eq(lo, tof_bounds[next_leg][0]) \
                and eq(hi, tof_bounds[next_leg][1]):
            tof_slots.append(i); next_leg += 1; continue      # ← tof
        if vinf_idx < 0 and (
                (eq(lo, vinf_mps[0]) and eq(hi, vinf_mps[1]))
                or (eq(lo, vinf_bounds[0]) and eq(hi, vinf_bounds[1]))):
            vinf_idx = i; continue                            # ← 发射 v∞
        if (eq(lo, -two_pi) and eq(hi, two_pi)) or \
                (eq(lo, -math.pi) and eq(hi, math.pi)):
            beta_slots.append(i); continue                    # ← beta
        if eq(lo, 0.1) and eq(hi, 0.9):
            n_alpha += 1; continue                            # ← alpha
        if 1.0 <= lo < hi <= 1000.0:
            rp_slots.append(i); continue                      # ← rp

    if not (t0_idx >= 0 and len(tof_slots) == n_legs
            and len(beta_slots) == n_legs - 1
            and len(rp_slots) == n_legs - 1
            and n_alpha == n_legs):                           # 数量必须严格对上
        raise LayoutError(
            "布局解析失败\n"
            f"  期望: t0=1, tof={n_legs}, beta={n_legs-1}, "
            f"rp={n_legs-1}, alpha={n_legs}\n"
            f"  实得: t0={'√' if t0_idx >= 0 else '×'}, "
            f"tof={len(tof_slots)}@{tof_slots}, "
            f"beta={len(beta_slots)}@{beta_slots}, "
            f"rp={len(rp_slots)}@{rp_slots}, alpha={n_alpha}\n"
            f"  完整边界: {list(zip(lb, ub))}")

    return {'t0_idx': t0_idx, 'vinf_idx': vinf_idx,
            'tof_slots': tof_slots, 'flyby_slots': list(zip(beta_slots, rp_slots))}


# ============== 动态包装 udp: 时长帽 + 计费口径 + 飞掠边界 ==============
class DynamicMGA_UDP:
    def __init__(self, seq, t0_bounds, tof_bounds, vinf_bounds, time_cap):
        self._udp = mga_1dsm(                                 # 主 udp: 全额计费基准
            seq=seq, t0=t0_bounds, tof=tof_bounds, vinf=vinf_bounds,
            add_vinf_dep=True, add_vinf_arr=True, multi_objective=False
        )
        self._seq_names = [p.name for p in seq]
        self._layout = resolve_layout(
            self._udp, t0_bounds, tof_bounds, vinf_bounds,
            add_vinf_dep=True, add_vinf_arr=True)
        self._tof_slots = self._layout['tof_slots']
        self._flyby_slots = self._layout['flyby_slots']
        self._time_cap = time_cap                             # 本序列专属时长帽
        self._resonant_legs = [                               # 相同行星相邻段 = 共振段
            i for i in range(len(self._seq_names) - 1)
            if self._seq_names[i] == self._seq_names[i + 1]
        ]
        # —— 计费探针: 同一决策变量, 仅 pykep 计费开关不同 → 差分剥离分量 ——
        #   f_full   = 出发v∞ + ΣDSM + 到达v∞   (主 udp)
        #   f_noarr  = 出发v∞ + ΣDSM            (到达不计费)
        #   f_novinf = ΣDSM                     (到达+出发都不计费)
        self._decompose_ok = False
        try:
            self._p_noarr = mga_1dsm(seq=seq, t0=t0_bounds, tof=tof_bounds,
                                     vinf=vinf_bounds, add_vinf_dep=True,
                                     add_vinf_arr=False, multi_objective=False)
            self._p_novinf = mga_1dsm(seq=seq, t0=t0_bounds, tof=tof_bounds,
                                      vinf=vinf_bounds, add_vinf_dep=False,
                                      add_vinf_arr=False, multi_objective=False)
            nx0 = pg.problem(self._udp).get_nx()              # 三者决策变量必须同构,
            nx1 = pg.problem(self._p_noarr).get_nx()          # 否则差分无意义
            nx2 = pg.problem(self._p_novinf).get_nx()
            self._decompose_ok = (nx0 == nx1 == nx2)
        except Exception:
            self._decompose_ok = False                        # 探针不可用→回退旧口径

    def __getattr__(self, name):
        if name.startswith('_'):
            raise AttributeError(name)                        # 内部成员不外透
        return getattr(self._udp, name)                       # 其余透传给底层 udp

    def fitness(self, x):
        # 1) 公平时长硬帽(每序列自己的帽子)
        if sum(x[s] for s in self._tof_slots) > self._time_cap:
            return [FEASIBILITY_PENALTY]
        f_full = float(self._udp.fitness(x)[0])               # dep + ΣDSM + arr
        total = f_full
        # 2) 任务口径: 到达超速按 JOI 捕获烧计费(差分剥离)
        if ARRIVAL_MODE == 'capture' and self._decompose_ok:
            f_noarr = float(self._p_noarr.fitness(x)[0])      # dep + ΣDSM
            arr = max(f_full - f_noarr, 0.0)                  # 到达超速(m/s)
            joi_ms = 1000.0 * joi_capture_dv_kms(arr / 1000.0)
            if CHARGE_DEPARTURE:
                total = f_noarr + joi_ms                      # 出发 + DSM + JOI
            else:
                dsm = float(self._p_novinf.fitness(x)[0])     # 纯 ΣDSM
                total = dsm + joi_ms                          # 出发完全交给火箭
        # 3) 共振约束(仅非 off 模式)
        if RESONANCE_MODE == 'soft':
            for i in self._resonant_legs:
                T = x[self._tof_slots[i]]
                period = PERIODS[self._seq_names[i]]
                dist = min(abs(T - k * period) - resonance_tol(k, period)
                           for k in RESONANCE_HARMONICS)
                if dist > 0:
                    total += RESONANCE_SOFT_WEIGHT * dist     # 按天软罚
        elif RESONANCE_MODE == 'hard':
            for i in self._resonant_legs:
                T = x[self._tof_slots[i]]
                period = PERIODS[self._seq_names[i]]
                if min(abs(T - k * period) - resonance_tol(k, period)
                       for k in RESONANCE_HARMONICS) > 0:
                    return [FEASIBILITY_PENALTY]              # 一票否决
        if TIME_WEIGHT > 0:                                   # 可选软时间罚
            total += TIME_WEIGHT * sum(x[s] for s in self._tof_slots) / 365.25
        return [total]

    def mission_breakdown(self, x):
        """→ 分解字典(km/s): 出发(火箭包)/ΣDSM/到达超速/JOI/航天器推进/总账"""
        try:
            if not self._decompose_ok:
                return None
            f_full = float(self._udp.fitness(x)[0])
            f_noarr = float(self._p_noarr.fitness(x)[0])
            f_novinf = float(self._p_novinf.fitness(x)[0])
            dep = max(f_noarr - f_novinf, 0.0) / 1000.0       # 出发超速
            dsm = f_novinf / 1000.0                           # 深空机动合计
            arr = max(f_full - f_noarr, 0.0) / 1000.0         # 到达超速
            joi = joi_capture_dv_kms(arr)                     # 捕获烧
            return {'dep': dep, 'dsm': dsm, 'arr': arr, 'joi': joi,
                    'sc': dsm + joi, 'total': dep + dsm + joi}
        except Exception:
            return None

    def get_bounds(self):
        lb, ub = self._udp.get_bounds()
        lb, ub = list(lb), list(ub)
        for i, planet_name in enumerate(self._seq_names[1:-1]):   # 逐飞掠行星收紧
            beta_bounds, rp_bounds = get_flyby_bounds(planet_name)
            b_idx, r_idx = self._flyby_slots[i]
            lb[b_idx], ub[b_idx] = beta_bounds                    # beta 按行星
            lb[r_idx], ub[r_idx] = rp_bounds                      # rp 按大气高度
        return lb, ub

    def pretty(self, x):
        return self._udp.pretty(x)

    def plot(self, x):
        return self._udp.plot(x)


# ============================ 序列 → 优化问题 ============================
def build_udp(seq_names):
    seq = [jpl_lp(name) for name in seq_names]
    t0_bounds = [epoch(T0_START_MJD2000), epoch(T0_END_MJD2000)]
    tof_bounds = []
    for i in range(len(seq_names) - 1):
        p1, p2 = seq_names[i], seq_names[i + 1]
        is_final = (i == len(seq_names) - 2)
        bounds = get_tof_bounds(p1, p2, is_final)             # 逐段动态包络
        if p1 == p2 and not is_final and RESONANCE_MODE == 'hard':
            narrowed = get_resonant_tof_bounds(p1, bounds)    # 共振剪枝仅 hard 模式
            if narrowed is not None:
                bounds = narrowed
        tof_bounds.append(bounds)
    cap = get_time_cap_days(tof_bounds)                       # 逐序列公平帽
    tof_bounds[-1][1] = min(tof_bounds[-1][1],                # 末段上界按帽收紧,
                            cap - sum(b[0] for b in tof_bounds[:-1]))  # 消死区
    vinf_bounds = get_vinf_bounds(seq_names)                  # 已按火箭能力封顶
    return DynamicMGA_UDP(seq, t0_bounds, tof_bounds, vinf_bounds, cap)


# ============================== 单序列求解器 ==============================
def _sade_once(prob, pop_size, seed):
    """一次完整 SADE: 返回 (最优fitness, 最优x)"""
    algo = pg.algorithm(pg.sade(gen=SADE_GEN, seed=seed))
    pop = pg.population(prob, size=pop_size, seed=seed)       # 种群按维度自适应
    pop = algo.evolve(pop)
    return float(pop.champion_f[0]), list(pop.champion_x)


def optimize_sequence_worker(task):
    """Pool worker: 多种子 SADE + COMPASS 精修 + 任务口径分解"""
    seq_names, seeds = task
    seq_str = seq_display(seq_names)
    try:
        udp = build_udp(seq_names)
        prob = pg.problem(udp)
        pop_size = pop_size_for(prob.get_nx())
        best_f1, best_x1 = math.inf, None
        for seed in seeds:                                    # 多种子独立重启
            f1, x1 = _sade_once(prob, pop_size, seed)
            if f1 < best_f1:
                best_f1, best_x1 = f1, x1
        if best_f1 >= INFEASIBLE_THRESHOLD:                   # 全败 → 追加种子重试
            for seed in (11, 12, 13):
                f1, x1 = _sade_once(prob, pop_size, seed)
                if f1 < best_f1:
                    best_f1, best_x1 = f1, x1
        best_f2, best_x2 = best_f1, best_x1
        try:
            pop2 = pg.population(prob)                        # 空种群注入 SADE 最优
            pop2.push_back(best_x1, prob.fitness(best_x1))
            algo2 = pg.algorithm(pg.compass_search(max_fevals=COMPASS_FEVALS))
            pop2 = algo2.evolve(pop2)                         # 坐标模式局部精修
            if float(pop2.champion_f[0]) < best_f1:
                best_f2 = float(pop2.champion_f[0])
                best_x2 = list(pop2.champion_x)
        except Exception as e:
            print(f"[WARN] {seq_str} COMPASS 精修失败: {e}", file=sys.stderr)
        bd = udp.mission_breakdown(best_x2)                   # 任务口径分解
        return seq_str, best_f1, best_f2, best_x2, seq_names, None, bd
    except LayoutError as e:                                  # 布局失败走错误通道
        return seq_str, FEASIBILITY_PENALTY, FEASIBILITY_PENALTY, None, \
            seq_names, f"布局解析失败: {e}", None
    except Exception as e:
        return seq_str, FEASIBILITY_PENALTY, FEASIBILITY_PENALTY, None, \
            seq_names, f"{type(e).__name__}: {e}", None


# ============================== 进度与汇总 ==============================
def report_progress(current, total, elapsed, seq_str, best_dv, failed=False):
    """单行进度条(tty)或逐行打印(重定向)"""
    note = " (失败)" if failed else ""
    if not sys.stdout.isatty():
        dv = f" | 当前最优: {best_dv:.2f} km/s" if best_dv is not None else ""
        print(f"[{current}/{total}] {seq_str}{note}{dv} | 已用 {elapsed:.0f}s",
              flush=True)
        return
    bar_len = 35
    filled = int(bar_len * current / total)
    bar = "█" * filled + "░" * (bar_len - filled)
    pct = current / total * 100
    eta_str = (f"{elapsed / current * (total - current):.0f}s"
               if current > 0 else "???")
    dv_str = f"| 当前最优: {best_dv:.2f} km/s" if best_dv is not None else ""
    line = (f"\r  [{bar}] {current}/{total} ({pct:.0f}%) "
            f"| 已用 {elapsed:.0f}s | 剩余 {eta_str} "
            f"| 刚完成: {seq_str}{note} {dv_str}")
    try:
        line = line[:os.get_terminal_size().columns - 5]      # 防止折行
    except Exception:
        line = line[:115]
    sys.stdout.write(line + " " * 5)
    sys.stdout.flush()


def run_pool(tasks, t_start):
    """进程池并行: 返回 (结果列表, 错误列表)"""
    results, errors = [], []
    num = min(cpu_count(), 8)                                 # 并行度封顶 8
    total = len(tasks)
    done = 0
    best_dv = None
    with Pool(processes=num) as pool:
        for result in pool.imap_unordered(optimize_sequence_worker, tasks):
            done += 1
            seq_str, f1, f2, x, names, err, bd = result
            elapsed = time.time() - t_start
            if err:                                           # 错误走独立通道
                errors.append((seq_str, err))
                report_progress(done, total, elapsed, seq_str, best_dv,
                                failed=True)
                continue
            results.append(result)
            dv = f2 * DV_TO_KMS
            if best_dv is None or dv < best_dv:
                best_dv = dv                                  # 实时刷新全局最优
            report_progress(done, total, elapsed, seq_str, best_dv)
    print()
    return results, errors


def status_of(f1, f2):
    """SADE→COMPASS 改善量体检"""
    if f1 >= INFEASIBLE_THRESHOLD or f2 >= INFEASIBLE_THRESHOLD:
        return "✗ 不可行/未找到可行解"
    diff = (f2 - f1) * DV_TO_KMS
    if abs(diff) > 0.5:
        return "⚠ 差距大"
    elif abs(diff) > 0.1:
        return "~ 有改善"
    return "✓ 收敛"


def merge_results(old, new):
    """按序列合并两批结果, 各保留历史最优"""
    best = {}
    for r in list(old) + list(new):
        k = tuple(r[4])
        if k not in best or r[2] < best[k][2]:
            best[k] = r
    return sorted(best.values(), key=lambda r: r[2])


# ================================= 主流程 =================================
def main():
    sequences = generate_sequences()
    total = len(sequences)

    print(f"共生成 {total} 种序列")
    print(f"发射窗口: {epoch(T0_START_MJD2000)} ~ {epoch(T0_END_MJD2000)} "
          f"(天问四备选窗 2028~2031)")
    print(f"借力次数: 0~4")
    print(f"时长包络: 每序列 min({FAIR_CAP_RATIO}×自身TOF下界和, "
          f"{FAIR_CAP_MAX_YEARS}年) | 共振模式: {RESONANCE_MODE}")
    cal = (f"出发v∞ + ΣDSM + JOI捕获(近木点{JOI_RP_RADII}R_J"
           f"→{JOI_PERIOD_DAYS:.0f}天椭圆)")
    who = ("出发v∞计入总ΔV, 报表单列可由火箭承担" if CHARGE_DEPARTURE
           else "出发v∞不计入(完全交给火箭)")
    print(f"ΔV 口径: 任务口径 = {cal} | {who}")
    print(f"扫描: 每序列 {len(SCAN_SEEDS)} seeds → 预算追加轮(≤{BUDGET_ROUNDS}) "
          f"→ Top-{TOP_N_REFINE} x {len(RESTART_SEEDS)} → 重点序列 x "
          f"{len(KEY_FEED_SEEDS)}")
    print(f"每序列: SADE({SADE_GEN}代) + COMPASS精修({COMPASS_FEVALS}次)\n")

    t_start = time.time()

    # ---------- 阶段一: 全序列统一预算扫描 ----------
    print("=== 阶段一: 全序列扫描 ===\n")
    scan_tasks = [(names, SCAN_SEEDS) for names in sequences]
    results, all_errors = run_pool(scan_tasks, t_start)
    refine_errors = []

    # ---------- 阶段一点五: 预算追加轮(欠收敛/未解出者加种子重赛) ----------
    already = set()                                           # 已追加过的不再重复
    for round_i in range(1, BUDGET_ROUNDS + 1):
        under = [r for r in results
                 if tuple(r[4]) not in already
                 and (r[2] >= INFEASIBLE_THRESHOLD
                      or (r[2] - r[1]) * DV_TO_KMS > BUDGET_GAP_KMS)]
        if not under:
            break
        print(f"\n=== 预算追加轮 {round_i}: {len(under)} 个欠收敛/未解出序列 "
              f"各 +{len(BUDGET_SEEDS)} seeds ===\n")
        more, err2 = run_pool([(r[4], BUDGET_SEEDS) for r in under],
                              time.time())
        refine_errors += [(f"[追加{round_i}]{s}", e) for s, e in err2]
        already.update(tuple(r[4]) for r in under)
        results = merge_results(results, more)

    # ---------- 阶段二: Top-N 多种子精算 ----------
    results.sort(key=lambda r: r[2])
    top = [r[4] for r in results[:TOP_N_REFINE]
           if r[2] < INFEASIBLE_THRESHOLD]
    if top:
        print(f"\n=== 阶段二: Top-{len(top)} 序列 "
              f"{len(RESTART_SEEDS)}-seed 重启精算 ===\n")
        refined, err3 = run_pool([(names, RESTART_SEEDS) for names in top],
                                 time.time())
        refine_errors += [(f"[精算]{s}", e) for s, e in err3]
        results = merge_results(results, refined)

    # ---------- 阶段三: 重点史实架构序列加喂(10 seeds) ----------
    print(f"\n=== 阶段三: 重点序列加喂 x {len(KEY_FEED_SEEDS)} seeds ===\n")
    key_more, err4 = run_pool([(list(s), KEY_FEED_SEEDS) for s in KEY_FEED],
                              time.time())
    refine_errors += [(f"[重点]{s}", e) for s, e in err4]
    results = merge_results(results, key_more)

    # ---------- 最终报表 ----------
    t_total = time.time() - t_start
    if refine_errors:
        print("\n" + "=" * 50)
        print(f"⚠ 捕获到 {len(refine_errors)} 个任务失败:")
        for s, e in refine_errors:
            print(f"  [{s}]: {e}")

    results.sort(key=lambda r: r[2])
    n_ok = sum(1 for r in results if r[2] < INFEASIBLE_THRESHOLD)
    print(f"\n总耗时: {t_total:.1f} 秒 ({t_total/60:.1f} 分钟) | "
          f"可行序列: {n_ok}/{total}")
    print("\n" + "=" * 110)
    print(f"          {total} 种序列 ΔV 排名表 (任务口径, 天问四备选窗 "
          f"2028~2031)")
    print("=" * 110)
    print(f"{'排名':<5} | {'序列':<30} | {'SADE':<12} | {'总ΔV':<12} "
          f"| {'出发(火箭)':<10} | {'航天器推进':<10} | {'状态'}")
    print("-" * 110)

    for rank, (seq_str, f1, f2, _, _, _, bd) in enumerate(results, 1):
        dv1_km, dv2_km = f1 * DV_TO_KMS, f2 * DV_TO_KMS
        if bd:
            dep_s, sc_s = f"{bd['dep']:.2f}", f"{bd['sc']:.2f}"
        else:
            dep_s, sc_s = "-", "-"
        print(f"{rank:<5} | {seq_str:<30} | {dv1_km:<12.3f} "
              f"| {dv2_km:<12.3f} | {dep_s:<10} | {sc_s:<10} "
              f"| {status_of(f1, f2)}")

    print("-" * 110)
    print("总ΔV = 出发v∞ + ΣDSM + JOI捕获;  '出发(火箭)'列可由运载火箭实现"
          f"(C3=v∞², 上限{DEPARTURE_VINF_MAX} km/s)")
    print("航天器推进 = ΣDSM + JOI = 探测器推进系统实际承担量; "
          "到达超速按 JOI 捕获烧计费(Oberth 效应)")

    # ---------- 冠军轨迹详情 ----------
    if results and results[0][3] is not None \
            and results[0][2] < INFEASIBLE_THRESHOLD:
        best = results[0]
        bd = best[6]
        print(f"\n{'=' * 50}")
        print(f"🏆 最佳序列: {best[0]}")
        if bd:
            print(f"    任务口径总ΔV = {bd['total']:.3f} km/s")
            print(f"    ├─ 出发超速 v∞ = {bd['dep']:.3f} km/s "
                  f"(C3 = {bd['dep'] ** 2:.2f} km²/s²)")
            print(f"    │    ↑ 可交给运载火箭实现: 发射时地球双曲线超速,")
            print(f"    │      火箭末级提供, 不消耗航天器推进剂")
            print(f"    └─ 航天器自担推进 = {bd['sc']:.3f} km/s")
            print(f"         = ΣDSM {bd['dsm']:.3f} + 木星捕获 JOI {bd['joi']:.3f}")
            print(f"    (到达超速 {bd['arr']:.3f} km/s 未全额计费, "
                  f"仅按 JOI 捕获烧 {bd['joi']:.3f} 计入)")
        try:
            udp = build_udp(best[4])
            T_years = sum(best[3][s] for s in udp._tof_slots) / 365.25
            print(f"    总飞行时长: {T_years:.1f} 年 (自身帽子 "
                  f"{udp._time_cap / 365.25:.1f} 年)")
            detail = udp.pretty(best[3])
            if isinstance(detail, str):
                print("\n" + detail)
        except Exception as e:
            print(f"    (轨迹细节输出失败: {e})")
        print(f"{'=' * 50}")


if __name__ == '__main__':
    main()
