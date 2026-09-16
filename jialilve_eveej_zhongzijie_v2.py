# -*- coding: utf-8 -*-
"""
Galileo E-V-E-E-J (VEEGA) 优化 —— v6: 两阶段法 + 正确的精修
============================================================
阶段一: 12 个履历参数钉死, 只搜 6 个方向参数 (sade 全局)
阶段二: 全 18 维精修:
        2a) compass_search 直接爬山 (替代缺失的 nelder_mead)
        2b) 以阶段一解为种子的全维 sade (fitness 好, 不会被淘汰)
"""

import math
import pygmo as pg
from pykep import epoch
from pykep.planet import jpl_lp
from pykep.trajopt import mga_1dsm


def make_base():
    """Galileo 真实履历编码的 18 维基底"""
    return [
        -3724.5,   # x[0]  t0 = 1989-10-18
        0.5, 0.5,  # x[1],x[2]  u,v 出发方向      (★待搜索)
        4000.0,    # x[3]  Vinf m/s              (★待搜索)
        0.3,       # x[4]  eta1 DSM 时刻
        115.0,     # x[5]  T1 E->V (真实 115 天)
        0.0,       # x[6]  beta2                 (★待搜索)
        2.6,       # x[7]  rp2 金星 (1.6万 km)
        0.5,       # x[8]  eta2
        301.0,     # x[9]  T2 V->E (真实 301 天)
        0.0,       # x[10] beta3                 (★待搜索)
        1.15,      # x[11] rp3 地球 960 km (低空)
        0.5,       # x[12] eta3
        730.0,     # x[13] T3 E->E 2:1 共振
        0.0,       # x[14] beta4                 (★待搜索)
        1.05,      # x[15] rp4 地球 303 km (超低空)
        0.4,       # x[16] eta4
        1094.0,    # x[17] T4 E->J (真实 1094 天)
    ]


class DirectionUDP:
    """6 维子问题: [u, v, Vinf, beta2, beta3, beta4], 其余钉死"""

    def __init__(self, prob, base):
        self._prob = prob
        self._base = list(base)

    def fitness(self, x):
        b = self._base[:]
        b[1], b[2] = x[0], x[1]
        b[3] = x[2]
        b[6], b[10], b[14] = x[3], x[4], x[5]
        return [self._prob.fitness(b)[0]]

    def get_bounds(self):
        p = math.pi
        return ([0.0, 0.0, 2500.0, -p, -p, -p],
                [1.0, 1.0, 5000.0,  p,  p,  p])


def report(udp, prob, best_x, seq):
    """统一输出"""
    f_best = prob.fitness(best_x)[0]
    print('\n================ 最终解 ================')
    print('Best ΔV = %.1f m/s (= %.3f km/s)' % (f_best, f_best / 1000))
    udp.pretty(best_x)

    rp = best_x[7::4]
    print('\n三次借力近心点 (行星半径):')
    print('  Venus   : %.2f R   (真实 ≈ 2.6)'  % rp[0])
    print('  Earth-1 : %.2f R   (真实 ≈ 1.15)' % rp[1])
    print('  Earth-2 : %.2f R   (真实 ≈ 1.05)' % rp[2])

    T = best_x[5::4]
    print('\n发射 V_inf : %.2f km/s' % (best_x[3] / 1000))
    print('发射历元   :', epoch(best_x[0]))
    t_acc = best_x[0]
    for i_, pl in enumerate(seq[1:]):
        t_acc += T[i_]
        print('到达 %-8s: %s  (该段 %6.1f 天 ≈ %.2f 年)' % (
            pl.name, epoch(t_acc), T[i_], T[i_] / 365.25))
    print('总飞行时间 : %.1f 天 ≈ %.2f 年' % (sum(T), sum(T) / 365.25))

    try:
        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        udp.plot(best_x)
        plt.title('Galileo VEEGA - total DV = %.2f km/s' % (f_best / 1000))
        plt.show()
    except Exception as e:
        print('绘图失败 (可忽略):', e)


def run():
    # ==================================================================
    # 1. 行星序列 + 原始 18 维问题
    # ==================================================================
    seq = [
        jpl_lp('earth'), jpl_lp('venus'),
        jpl_lp('earth'), jpl_lp('earth'), jpl_lp('jupiter'),
    ]
    Y = 365.25
    udp = mga_1dsm(
        seq=seq,
        t0=[epoch(-3731.5), epoch(-3717.5)],   # 1989-10-11 ~ 10-25
        tof=[
            [0.25 * Y, 0.50 * Y],   # E -> V
            [0.65 * Y, 0.95 * Y],   # V -> E
            [1.80 * Y, 2.20 * Y],   # E -> E 2:1 共振
            [2.50 * Y, 3.50 * Y],   # E -> J
        ],
        vinf=[2.5, 5.0],
        add_vinf_dep=True,
        add_vinf_arr=True,
        multi_objective=False,
    )
    prob = pg.problem(udp)

    # ==================================================================
    # 2. 阶段一: 6 维方向参数全局搜索 (多种子)
    # ==================================================================
    base = make_base()
    rprob = pg.problem(DirectionUDP(prob, base))
    print('阶段一: 只搜 6 个方向参数 ...')
    best_f, best6 = None, None
    for s in range(1, 4):
        pop = pg.population(rprob, 50, seed=s)
        pop = pg.algorithm(pg.sade(gen=100)).evolve(pop)
        print('  seed %d: ΔV = %.1f m/s' % (s, pop.champion_f[0]))
        if best_f is None or pop.champion_f[0] < best_f:
            best_f, best6 = pop.champion_f[0], pop.champion_x

    full = list(base)
    full[1], full[2] = best6[0], best6[1]
    full[3] = best6[2]
    full[6], full[10], full[14] = best6[3], best6[4], best6[5]
    print('阶段一最优: ΔV = %.1f m/s' % prob.fitness(full)[0])

    # ==================================================================
    # 3. 阶段 2a: 全 18 维 compass_search 爬山 (替代缺失的 nelder_mead)
    #    compass_search: 坐标轴方向模式搜索, 无需梯度, 精修足够用
    # ==================================================================
    print('\n阶段 2a: compass_search 全维精修 ...')
    try:
        pop = pg.population(prob)
        try:
            pop.push_back(full)              # pygmo 2.x 自动算 fitness
        except TypeError:
            pop.push_back(full, list(prob.fitness(full)))
        # 最大评估数, 初始步长 (以搜索空间尺度的一定比例), 步长收缩系数
        algo = pg.algorithm(pg.compass_search(max_fevals=5000,
                                              start_range=0.1))
        pop = algo.evolve(pop)
        print('  精修后: ΔV = %.1f m/s' % pop.champion_f[0])
        if pop.champion_f[0] < prob.fitness(full)[0]:
            full = list(pop.champion_x)
    except Exception as e:
        print('  compass_search 失败:', e)

    # ==================================================================
    # 4. 阶段 2b: 以阶段一/2a 的好解为种子, 全 18 维 sade 再进化
    #    此时种子 fitness (~11 km/s) 远优于随机个体 (~20+ km/s),
    #    不会被淘汰, 进化器会在其邻域内继续开采
    # ==================================================================
    print('\n阶段 2b: 全维 sade (好解为种子) ...')
    try:
        pop = pg.population(prob, 29, seed=1)
        try:
            pop.push_back(full)
        except TypeError:
            pop.push_back(full, list(prob.fitness(full)))
        pop = pg.algorithm(pg.sade(gen=800)).evolve(pop)
        print('  精修后: ΔV = %.1f m/s' % pop.champion_f[0])
        if pop.champion_f[0] < prob.fitness(full)[0]:
            full = list(pop.champion_x)
    except Exception as e:
        print('  阶段 2b 失败:', e)

    # ==================================================================
    # 5. 输出 + 绘图
    # ==================================================================
    report(udp, prob, full, seq)


if __name__ == '__main__':
    run()
