# -*- coding: utf-8 -*-
"""
Galileo 复现: E-V-E-E-J (VEEGA) 多重力辅助转移轨道优化 —— v3
================================================================
v3 改进:
1. vinf=[2.5, 5.0]: 抬高出发 v_inf 下限, 把优化器从
   "低发射能量 + 大DSM" 的局部最优家族逼进 "高能量发射 + 贴近借力"
   的正确家族 (真实 Galileo 的做法)
2. 多种子循环取最优, 进一步避开局部最优
3. 各段 tof 按 Galileo 真实履历锁定 (2:1 共振保留)
"""

import pygmo as pg
from pykep import epoch
from pykep.planet import jpl_lp
from pykep.trajopt import mga_1dsm


def run():
    # ==================================================================
    # 1. 行星序列: Earth -> Venus -> Earth -> Earth -> Jupiter (VEEGA)
    # ==================================================================
    seq = [
        jpl_lp('earth'),
        jpl_lp('venus'),
        jpl_lp('earth'),
        jpl_lp('earth'),
        jpl_lp('jupiter'),
    ]

    # ==================================================================
    # 2. UDP —— 与 v2 相同, 只改 vinf
    # ==================================================================
    Y = 365.25
    udp = mga_1dsm(
        seq=seq,
        t0=[epoch(-3731.5), epoch(-3717.5)],   # 1989-10-11 ~ 10-25
        tof=[
            [0.25 * Y, 0.50 * Y],   # E -> V  (真实 115 天)
            [0.65 * Y, 0.95 * Y],   # V -> E  (真实 ~300 天)
            [1.80 * Y, 2.20 * Y],   # E -> E  2:1 共振 (真实 730 天)
            [2.50 * Y, 3.50 * Y],   # E -> J  (真实 ~3 年)
        ],
        # ★ 关键改动: 出发 v_inf 强制在 2.5~5.0 km/s
        #   (即 C3 ≈ 6~25 km²/s², 类似真实 Galileo 的 IUS 发射能量)
        vinf=[2.5, 5.0],
        add_vinf_dep=True,
        add_vinf_arr=True,
        multi_objective=False,
    )
    prob = pg.problem(udp)
    print(prob)

    # ==================================================================
    # 3. 多种子循环: 每个种子跑一次群岛, 取全局最优
    #    MGA-1DSM 多局部极值, 多启动是标配手段
    # ==================================================================
    n_seeds = 3
    best_f, best_x, best_seed = None, None, None

    for seed in range(1, n_seeds + 1):
        archi = pg.archipelago(
            n=10,
            algo=pg.sade(gen=100),
            prob=prob,
            pop_size=30,
            seed=seed,
        )
        print('Running sade, seed = %d ...' % seed)
        archi.evolve()
        archi.wait()

        fs = archi.get_champions_f()
        xs = archi.get_champions_x()
        i = fs.index(min(fs))
        print('  seed %d: best ΔV = %.1f m/s' % (seed, min(fs)[0]))

        if best_f is None or min(fs)[0] < best_f:
            best_f, best_x, best_seed = min(fs)[0], xs[i], seed

    # ==================================================================
    # 4. 输出全局最优
    # ==================================================================
    print('\n================ 全局最优 (seed=%d) ================' % best_seed)
    print('Best ΔV = %.1f m/s (= %.3f km/s)' % (best_f, best_f / 1000))
    udp.pretty(best_x)

    # 关键指标核对: 借力近心点应显著变小 (真实 Galileo: 金 3.6R, 地 1.05R)
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
