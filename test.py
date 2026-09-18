# -*- coding: utf-8 -*-
"""
test.py —— mga_scan.py 的部署自检脚本 (原 check_layout)
三部分:
  【1】单序列详细布局 dump (E-V-E-J, 即上次失败的那条)
  【2】全部 40 序列冒烟测试 (构建/边界/fitness/共振角点一致性)
  【3】E-V-E-J 小规模 SADE 试跑, 验证优化链路与 ΔV 量纲
预期几秒~几十秒完成。全部通过后再运行 mga_scan.py。
注意: 文件名 test.py 若与 pytest 冲突, 可改名, 只需保持 import mga_scan 可用。
"""

import math
import pygmo as pg

from mga_scan import (generate_sequences, build_udp, seq_display, pick_champion,
                      PERIODS, RESONANCE_HARMONICS, resonance_tol,
                      FEASIBILITY_PENALTY, INFEASIBLE_THRESHOLD, DV_TO_KMS)

FAILURES = []


def fail(msg):
    FAILURES.append(msg)
    print(f"  ✗ {msg}")


# ==================== 【1】单序列布局 dump ====================

def dump_layout():
    print("=" * 64)
    print("【1】单序列布局解析: E -> V -> E -> J\n")
    names = ['earth', 'venus', 'earth', 'jupiter']
    udp = build_udp(names)
    prob = pg.problem(udp)
    lb, ub = prob.get_bounds()
    L = udp._layout

    labels = ['?'] * len(lb)
    labels[L['t0_idx']] = 't0'
    if L.get('vinf_idx', -1) >= 0:
        labels[L['vinf_idx']] = 'vinf_dep'
    for leg_i, slot in enumerate(L['tof_slots']):
        labels[slot] = f'T{leg_i + 1}'
    for enc_i, (b, r) in enumerate(L['flyby_slots']):
        labels[b] = f'beta{enc_i + 1}'
        labels[r] = f'rp{enc_i + 1}'
    for i in range(len(lb)):
        if labels[i] == '?':
            if abs(lb[i] - 0.1) < 1e-9 and abs(ub[i] - 0.9) < 1e-9:
                labels[i] = 'alpha'
            else:
                labels[i] = '内部槽(透传)'

    print(f"  维度: {len(lb)}")
    for i, (l, u) in enumerate(zip(lb, ub)):
        print(f"    x[{i:>2}] {labels[i]:<12} [{l:>12.4f}, {u:>12.4f}]")
    print(f"\n  布局字典: {L}")
    print("  ✓ 布局解析通过\n")


# ==================== 【2】全序列冒烟测试 ====================

def corner_consistency(udp):
    """共振段 tof 界角点必须落在共振窗口内(验证 #8 收窄与 fitness 预检一致)"""
    lb, ub = udp.get_bounds()
    for i in udp._resonant_legs:
        period = PERIODS[udp._seq_names[i]]
        for corner in (lb, ub):
            T = corner[udp._tof_slots[i]]
            if not any(abs(T - k * period) <= resonance_tol(k, period)
                       for k in RESONANCE_HARMONICS):
                return False, (udp._seq_names[i], T)
    return True, None


def smoke_all():
    print("=" * 64)
    print("【2】全部序列冒烟测试\n")
    sequences = generate_sequences()
    passed = 0
    for names in sequences:
        tag = seq_display(names)
        try:
            udp = build_udp(names)
            prob = pg.problem(udp)
            lb, ub = prob.get_bounds()

            assert all(math.isfinite(v) for v in lb), "下界含非有限值"
            assert all(math.isfinite(v) for v in ub), "上界含非有限值"
            assert all(u >= l for l, u in zip(lb, ub)), "存在 ub < lb"

            x_mid = [(l + u) / 2.0 for l, u in zip(lb, ub)]
            f = float(prob.fitness(x_mid)[0])
            assert math.isfinite(f), "fitness 返回非有限值"

            note = ""
            if f < INFEASIBLE_THRESHOLD:
                ok, info = corner_consistency(udp)
                if not ok:
                    raise AssertionError(
                        f"共振角点越界: {info[0]} T={info[1]:.2f}d")

            passed += 1
            print(f"  ✓ {tag:<30} 维度={len(lb):>2}  tof槽位="
                  f"{udp._layout['tof_slots']}{note}")
        except Exception as e:
            fail(f"{tag}: {type(e).__name__}: {str(e)[:100]}")

    print(f"\n  通过: {passed}/{len(sequences)}\n")


# ==================== 【3】小规模试跑(链路+量纲验证) ====================

def mini_optimize():
    print("=" * 64)
    print("【3】E -> V -> E -> J 小规模试跑 (SADE 100代, 验证链路与量纲)\n")
    names = ['earth', 'venus', 'earth', 'jupiter']
    udp = build_udp(names)
    prob = pg.problem(udp)
    archi = pg.archipelago(n=1, algo=pg.sade(gen=100, seed=7),
                           prob=prob, pop_size=15, seed=7)
    archi.evolve()
    archi.wait()
    _, f = pick_champion(archi.get_champions_f(), archi.get_champions_x())
    dv = f * DV_TO_KMS
    print(f"  试跑最优 fitness = {f:.1f} (原始单位) = {dv:.3f} km/s")
    if f >= INFEASIBLE_THRESHOLD:
        print("  ⚠ 小种群未找到可行解(未必异常), 但请通过【2】确认共振窗口可行")
    elif 0.1 < dv < 30:
        print("  ✓ ΔV 处于合理量级(个位数 km/s), 量纲正常")
    else:
        print(f"  ⚠ ΔV 量纲可疑: 若显示约 1000 倍偏差, "
              f"请把 mga_scan.py 中 DV_TO_KMS 改为 1.0")
    print()


def main():
    try:
        dump_layout()
    except Exception as e:
        print(f"  ✗ 布局解析失败: {e}")
        print("\n→ 请把上面的完整报错(含边界 dump)发给审计方定位布局")
        raise SystemExit(1)
    smoke_all()
    mini_optimize()
    print("=" * 64)
    if FAILURES:
        print(f"自检结束: {len(FAILURES)} 项失败, 详见上方 ✗ 条目, 不要运行 mga_scan.py")
        raise SystemExit(1)
    print("自检全部通过, 可运行: python mga_scan.py")


if __name__ == '__main__':
    main()
