# -*- coding: utf-8 -*-
"""
Galileo 复现: E-V-E-E-J (VEEGA) 多重力辅助转移轨道优化 —— v4 (种子注入版)
================================================================================
核心改进: 把 Galileo 真实飞行履历编码为初始种子解, 注入每个初始种群。
正确解 (低空借力 + 倾斜共振轨道) 在参数空间里是一根"针",
纯随机初始化的进化算法几乎不可能撞进去, 必须给种子。
"""

import pygmo as pg
from pykep import epoch
from pykep.planet import jpl_lp
from pykep.trajopt import mga_1dsm


def make_x0():
    """按 Galileo 真实履历构造 18 维种子染色体。

    染色体布局 (direct 编码):
      x[0]          : 发射历元 (MJD2000)
      x[1], x[2]    : 出发 v_inf 方向参数 (u,v ∈ [0,1], 先给中性值)
      x[3]          : 出发 v_inf 大小
      x[4]          : 第1段 DSM 时刻比例 eta1
      x[5]          : 第1段飞行时间 T1
      之后每段 4 个 : [beta 借力面角, rp/rV 借力近心点半径比, eta, T]
    """
    x0 = [
        # ---- 出发段 ----
        -3724.5,          # t0 = 1989-10-18 (Galileo 真实发射日)
        0.5, 0.5,         # v_inf 方向参数 (未知, 给中间值让进化器调)
        4000.0,           # v_inf = 4.0 km/s (高能金星转移, 115 天到达)

        # ---- leg1: E -> V (真实 115 天) ----
        0.3,              # eta1
        115.0,            # T1

        # ---- leg2: V -> E (借力1, 真实 1990-02-10, 高度 ~1万 km) ----
        0.0,              # beta (未知, 给 0)
        2.6,              # rp/rV ≈ 16000 km / 6052 km  (金星真实借力)
        0.5,              # eta2
        301.0,            # T2 (1990-02-10 -> 1990-12-08)

        # ---- leg3: E -> E (借力2, 真实 1990-12-08, 高度 960 km) ----
        0.0,
        1.15,             # rp/rV ≈ (6378+960)/6378  (低空! 关键)
        0.5,
        730.0,            # T3: 2:1 共振, 整 2 年

        # ---- leg4: E -> J (借力3, 真实 1992-12-08, 高度 303 km) ----
        0.0,
        1.05,             # rp/rV ≈ (6378+303)/6378  (超低空! 最关键)
        0.4,
        1094.0,           # T4 (1992-12-08 -> 1995-12-07)
    ]
    return x0


def run():
    # ==================================================================
    # 1. 行星序列: Earth -> Venus -> Earth -> Earth -> Jupiter
    # ==================================================================
    seq = [
        jpl_lp('earth'),
        jpl_lp('venus'),
        jpl_lp('earth'),
        jpl_lp('earth'),
        jpl_lp('jupiter'),
    ]

    # ==================================================================
    # 2. UDP (与 v3 相同的窗口, 按 Galileo 履历锁定)
    # ==================================================================
    Y = 365.25
    udp = mga_1dsm(
        seq=seq,
        t0=[epoch(-3731.5), epoch(-3717.5)],   # 1989-10-11 ~ 10-25
        tof=[
            [0.25 * Y, 0.50 * Y],   # E -> V  (真实 115 天)
            [0.65 * Y, 0.95 * Y],   # V -> E  (真实 301 天)
            [1.80 * Y, 2.20 * Y],   # E -> E  2:1 共振 (真实 730 天)
            [2.50 * Y, 3.50 * Y],   # E -> J  (真实 1094 天)
        ],
        vinf=[2.5, 5.0],            # 高能发射家族
        add_vinf_dep=True,
        add_vinf_arr=True,
        multi_objective=False,
    )
    prob = pg.problem(udp)
    print(prob)

    # ==================================================================
    # 3. 多种子进化, 每个种群里注入 Galileo 种子解
    #    种子 + 29 个随机个体: 既保留全局探索, 又保证落在正确盆地附近
    # ==================================================================
    x0 = make_x0()
    n_seeds = 3
    best_f, best_x, best_seed = None, None, None

    for seed in range(1, n_seeds + 1):
        # 29 个随机个体
        pop = pg.population(prob, 29, seed=seed)
        # 注入 Galileo 种子解 (第 30 个个体)
        try:
            pop.push_back(x0)                       # 新版 pygmo 自动算 fitness
        except TypeError:
            pop.push_back(x0, list(prob.fitness(x0)))  # 旧版需手动传 fitness

        algo = pg.algorithm(pg.sade(gen=100))
        print('Running sade (seed=%d, 29 random + 1 Galileo seed) ...' % seed)
        pop = algo.evolve(pop)

        f = pop.champion_f[0]
        print('  seed %d: best ΔV = %.1f m/s' % (seed, f))
        if best_f is None or f < best_f:
            best_f, best_x, best_seed = f, pop.champion_x, seed

    # ==================================================================
    # 4. 输出全局最优
    # ==================================================================
    print('\n================ 全局最优 (seed=%d) ================' % best_seed)
    print('Best ΔV = %.1f m/s (= %.3f km/s)' % (best_f, best_f / 1000))
    udp.pretty(best_x)

    # ---- 核对关键指标: 借力半径是否降到低位 ----
    # 染色体中 rp/rV 位于 7, 11, 15 (三次借力)
    rp = best_x[7::4]
    print('\n三次借力近心点 (行星半径):')
    print('  Venus   : %.2f R   (真实 Galileo ≈ 2.6)'  % rp[0])
    print('  Earth-1 : %.2f R   (真实 Galileo ≈ 1.15)' % rp[1])
    print('  Earth-2 : %.2f R   (真实 Galileo ≈ 1.05)' % rp[2])

    T = best_x[5::4]
    print('\n发射 V_inf : %.2f km/s' % (best_x[3] / 1000))
    print('发射历元   :', epoch(best_x[0]))
    t_acc = best_x[0]
    for i_, pl in enumerate(seq[1:]):
        t_acc += T[i_]
        print('到达 %-8s: %s  (该段 %6.1f 天 ≈ %.2f 年)' % (
            pl.name, epoch(t_acc), T[i_], T[i_] / 365.25))
    print('总飞行时间 : %.1f 天 ≈ %.2f 年' % (sum(T), sum(T) / 365.25))

    # ==================================================================
    # 5. 绘图
    # ==================================================================
    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        udp.plot(best_x)
        plt.title('Galileo VEEGA - total DV = %.2f km/s' % (best_f / 1000))
        plt.show()
    except Exception as e:
        print('绘图失败 (可忽略):', e)


if __name__ == '__main__':
    run()
