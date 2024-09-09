#!/usr/bin/env python
"""
azsim.py
"""
import argparse
import csv
import math
import random
import sys

DBG = 2

def dump_list(wp, headers, vals):
    """
    dump a list, with leading column=name
    """
    for i, x in enumerate(headers + [f"{x}" for x in vals]):
        if i == 0:
            wp.write(f"{x}")
        else:
            wp.write(f",{x}")
    wp.write("\n")

def very_close(a, b):
    """
    a approx b
    """
    return abs(a - b) < 1.0e-6

def round_list(l, d):
    """
    round each element of l to d-digits after a decimal point,
    just for logging/debugging purposes
    """
    return [round(x, d) for x in l]

def round_int_float_list(l, d):
    """
    round each element of l to d-digits after a decimal point,
    just for logging/debugging purposes
    """
    return [(i, round(x, d)) for i, x in l]

def first_non_empty(l):
    for i, x in enumerate(l):
        if x != "":
            return i
    return len(l)
    
class Subsc:
    """
    サブスク

    変化する状態
    weight : 重み (毎期変わりうる)
    used : その期の使用量
    pend : 超過分で支払い未確定のもの(負債); このあとチャラになる可能性あり

    履歴
    hist_weigh : 期ごとの重み
    hist_used : 期ごとの使用クレ量
    hist_free : 期ごとの無料クレ量
    hist_pend : 期ごとのたまった超過分で支払い未確定のもの
    hist_disc : 期ごと割引でされたクレ量
    hist_paid : 期ごとの支払ったクレ量
    """
    def __init__(self, subsc_id, t_idx):
        # ID (0, 1, 2, ...)
        self.subsc_id = subsc_id
        # ------- 状態 -------
        # 現在の重み
        self.weight = 1.0
        # その期に使用したクレジット
        self.used = 0.0
        # 支払い未確定の超過クレ(借り)
        self.pend = 0.0
        # ------- 履歴 -------
        # 各期の重み
        self.hist_weight = [""] * t_idx
        # 各期の使ったクレ
        self.hist_used = [""] * t_idx
        # 各期で無料(免除)になったクレ
        self.hist_free = [""] * t_idx
        # 各期で超過分, 支払い未確定クレ
        self.hist_pend = [""] * t_idx
        # 各期で無料(免除)になったクレ
        self.hist_disc = [""] * t_idx
        # 各期で支払ったクレ
        self.hist_paid = [""] * t_idx
        # 各期のS値
        self.hist_S = [""] * t_idx

    def dump_history(self, wp):
        """
        dump history to wp
        """
        sid = self.subsc_id
        dump_list(wp, [f"{sid}", "weight"], self.hist_weight)
        dump_list(wp, [f"{sid}", "S"],      self.hist_S)
        dump_list(wp, [f"{sid}", "used"],   self.hist_used)
        dump_list(wp, [f"{sid}", "free"],   self.hist_free)
        dump_list(wp, [f"{sid}", "pend"],   self.hist_pend)
        dump_list(wp, [f"{sid}", "disc"],   self.hist_disc)
        dump_list(wp, [f"{sid}", "paid"],   self.hist_paid)

    def plot_use_bars(self, ax, title, periods):
        """
        plot used, pending, discounted, and paid credits with bars
        """
        s = first_non_empty(self.hist_used)
        n_periods = len(self.hist_used[s:])
        bars = [("free", self.hist_free[s:]),
                ("pend", self.hist_pend[s:]),
                ("disc", self.hist_disc[s:]),
                ("paid", self.hist_paid[s:])]
        bottom = [0.0] * n_periods
        for label, data in bars:
            ax.bar(periods[s:], data, bottom=bottom, label=label)
            bottom = [x + y for x, y in zip(bottom, data)]
        lines = [("used", "black", self.hist_used[s:]),
                 ("S",    "yellow", self.hist_S[s:])]
        # [0,1,2,3] ->
        # [0,1,1,2,2,3,3,4] - 0.5
        if n_periods > 0:
            last_period = periods[-1]
            periods_ = [x - 0.5 for x in (periods[s:] + [last_period + 1]) for _ in range(2)][1:-1]
        else:
            periods_ = []
        for label, color, data in lines:
            # convert data like
            #    [0,1,2,3]
            #    [a,b,c,d]
            # -> [0,1,1,2,2,3,3,4]
            #    [a,a,b,b,c,c,d,d]
            data_    = [x for x in data    for _ in range(2)]
            ax.plot(periods_, data_, label=label, lw=2, color=color)
        ax.set_ylim(bottom=0.0)
        ax.set_title(title)
        ax.legend()

    def receive_free_credit(self, t_idx, t, k):
        """
        各期の終わりの, 無料クレジット受取り操作
        最大 k * weight まで「無料」
        """
        x = self.used + self.pend
        weight = self.weight
        kw = k * weight
        if x < kw:
            waived = x
        else:
            waived = kw
        new_pend = x - waived
        if DBG>=2:
            print(f"t {t_idx} {t} subsc={self.subsc_id} waive_credit : new pend"
                  f" = used [{self.used:.2f}] + pend [{self.pend:.2f}]"
                  f" - waived (up to) [{kw:.2f}] -> {new_pend:.2f}")
        assert(new_pend >= 0.0), (x, k, weight)
        self.pend = new_pend
        self.used = 0.0
        self.hist_free.append(waived)
        self.hist_pend.append(new_pend)
        return waived

    def receive_discount_credit(self, t_idx, t, k, do_settle):
        """
        精算期(e.g., 半年ごととか)の割引クレジット受取り
        """
        assert(self.used == 0.0), self.used
        if do_settle:
            x = self.pend
            weight = self.weight
            kw2 = k * weight / 2.0
            if x < k * weight:
                discounted = x - x ** 2 / (2.0 * k * weight)
            else:
                discounted = kw2
            paid = x - discounted
            if DBG>=2:
                print(f"t {t_idx} {t} subsc={self.subsc_id} waive_credit : paid"
                      f" = pend [{self.pend:.2f}]"
                      f" - discounted (up to) [{kw2:.2f}] -> {paid:.2f}")
            self.pend = 0.0
        else:
            if DBG>=2:
                print(f"t {t_idx} {t} subsc={self.subsc_id} receive_discount_credit :"
                      " do not settle in this period")
            discounted = 0.0
            paid = 0.0
        self.hist_disc.append(discounted)
        self.hist_paid.append(paid)
        return discounted

    def F0(self, k):
        x = self.used + self.pend
        return min(x, k * self.weight)
    
    def F1(self, k):
        if k == 0.0:
            return 0.0
        else:
            x = self.pend
            kw = k * self.weight
            if x < kw:
                return x - x * x / (2.0 * kw)
            else:
                return kw / 2.0
    
    def record_S(self, t_idx, t, F0, W):
        """
        record S
        """
        S = F0 * self.weight / W
        assert(S >= 0.0), (F0, self.weight, W)
        if DBG>=2:
            print(f"t {t_idx} {t} subsc={self.subsc_id} : calc S-value"
                  f" S = F0 [{F0:.2f}] * weight [{self.weight:.2f}] / W [{W:.2f}] -> S [{S:.2f}]")
        self.hist_S.append(S)

def gen_lognormal(rg, mu, sigma):
    """
    draw a sample from lognormal distribution
    """
    normal_sample = rg.gauss(mu, sigma)
    # Step 2: Exponentiate the normal sample to get a log-normal distribution
    lognormal_sample = math.exp(normal_sample)
    return lognormal_sample

USER_ALGO_C0 = "C0"
USER_ALGO_S  = "S"

def parse_num(s, empty_val):
    """
    parse a number
    """
    if s == "":
        return empty_val
    else:
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            pass
        return s

def parse_args(argv):
    """
    parse command line args
    """
    aa = argparse.ArgumentParser
    psr = aa(prog="azsim.py",
             description="simulate Azure Credit consumption")
    pa = psr.add_argument
    pa("--script", metavar="SCRIPT.CSV",
       help="simulate this CSV file")
    pa("--hist", metavar="HIST.CSV",
       help="write results to this CSV file")
    pa("--C0", metavar="INITIAL_CREDIT",
       help="initial credit (= credit to consume after the whole periods)")
    pa("--n-budget-periods", metavar="PERIODS",
       help="number of periods in which to consume C0")
    pa("--period", metavar="T0,T1,T2,...",
       help="period (see also --start-t and --end-t)")
    pa("--start-t", metavar="T", help="start at period T")
    pa("--end-t", metavar="T", help="end at period T")
    pa("--settle-period", metavar="T0,T1,T2,...",
       help=("settle payment at period T0, T1, T2,"
             " ... (see also --settle-interval)"))
    pa("--settle-interval", metavar="dT",
       help=("settle payment at every dT periods"
             " (i.e., at dT, 2dT, 3dT, ...)"))
    pa("--init-subscs", metavar="N",
       help="start with N subscriptions")
    pa("--mean-delta-subscs", metavar="N",
       help=("on average, add N subscs in a period"
             " (0 means no subscs are added)"))
    pa("--weight-change-prob", metavar="P",
       help=("each subsc changes weight in each period with"
             " probability P (0 means never)"))
    pa("--user-algo", metavar="ALGO",
       help=("choose user's behavior (S : use proportionally to the S-value,"
             " C0 : use proportionally to C0/n)"))
    pa("--mean-use-factor", metavar="F",
       help=("on average, use F * V in each period,"
             " where V is determined by C0, N, ALGO, subsc weights"))
    pa("--random-use", metavar="0/1",
       help="whether or not randomize credit usage (1 means yes)")
    pa("--seed", metavar="S",
       help="do not use any randomness to determine how much a user uses")
    pa("--dbg", metavar="LEVEL",
       help="debug output level")
    pa("--plots", metavar="S0,S1,...",
       help="plot overall consumption and histories of subsc S0, S1, ... at the end")
    opt = psr.parse_args(argv[1:])
    return opt

def parse_opt_plots(plots):
    """
    parse --plots option
    """
    if plots.lower() == "none":
        # do not show graphs at all
        return None
    else:
        plots = [s.strip() for s in plots.split(",")]
        return plots

def parse_opt_period_list(period_list):
    """
    parse --period and --settle-period
    """
    return [s.strip() for s in period_list.split(",")]

def date_str(idx):
    """
    idx = 1 <-> 2024/10
    idx = 2 <-> 2024/11
    idx = 3 <-> 2024/12
    idx = 4 <-> 2025/01
      ...
    """
    month_idx = (idx + 8) % 12
    year = (idx + 8) // 12 + 2024
    return f"{year}/{month_idx+1}"

def parse_opt_and_set_defaults(opt, script):
    """
    parse options and set default values
    """
    O = vars(opt)
    defaults = [
        ("script",             None,       str),
        ("hist",               "hist.csv", str),
        ("C0",                 1.5e8,      float),
        ("n_budget_periods",   60,         int),
        ("period",             None,       parse_opt_period_list),
        ("start_t",            1,          int),
        ("end_t",              None,       str),
        ("settle_period",      None,       parse_opt_period_list),
        ("settle_interval",    6,          int),
        ("init_subscs",        10,         int),
        ("mean_delta_subscs",  0.0,        float),
        ("weight_change_prob", 0.0,        float),
        ("user_algo",          "C0",       str),
        ("mean_use_factor",    1.0,        float),
        ("random_use",         1,          int),
        ("seed",               12345,      int),
        ("plots",              [0,1],      parse_opt_plots),
        ("dbg",                2,          int),
    ]
    # set the default value for options set by neither command line
    # nor script set the value
    for k, v, p in defaults:
        if O[k] is None:
            O[k] = v
        else:
            O[k] = p(O[k])
    # some special ad-hoc handlings
    if opt.end_t is None:
        # end_t の指定なし
        if script is not None:
            # scriptの場合, ファイルにusedが書いてある列まで
            len_used = max(len(sc["used"]) for sc in script.values())
            opt.end_t = opt.start_t + len_used - 1
        else:
            # そうでなければ予算の終わりまで
            opt.end_t = opt.n_budget_periods
    else:
        # end_t の指定有り. 数字として読めるかどうかを調べる
        opt.end_t = parse_num(opt.end_t, opt.end_t)
    if opt.period is None:
        # period が指定なし
        if isinstance(opt.end_t, type(0)):
            # end_t が数字の場合 -> start_t と end_t から生成
            end_t = opt.end_t
            opt.period = [date_str(i) for i in range(opt.start_t, end_t + 1)]
        else:
            # end_t が数字じゃない場合 -> エラー
            print(f"couldn't parse --end-t ({opt.end_t}) as an integer")
            print("either specify '--period' as a list of symbols"
                  " or '--end-t' as an index to the periods")
            return 0        # NG
    elif opt.end_t in opt.period:
        # period 指定有り
        # end_t が period に含まれていればそこまで
        end_idx = opt.period.index(opt.end_t)
        opt.period = opt.period[:end_idx + 1]
    else:
        # end_t が period に含まれていない
        # end_t が数字ならばそのindexまで
        end_idx = parse_num(opt.end_t, None)
        if end_idx is not None:
            # opt.period[0] <=> globalの start_idx
            # opt.period[?] <=> globalの end_idx
            opt.period = opt.period[:end_idx - start_idx + 1]
        else:
            print(f"couldn't find --end-t ({opt.end_t}) in --period list")
            return 0        # NG
    if opt.settle_period is None:
        # say settle_period is 6, then we settle on
        # 6, 12, 18, ...
        # if start_t = 4, then, they are period[2], period[8], ..
        # t=4       t=5       t=6            t=end_t          
        # period[0] period[1] period[2]  ... period[end_t-4]
        # start_idx = opt.settle_interval - opt.start_t % opt.settle_interval
        # settle_idxs = list(range(start_idx, opt.end_t + 1, opt.settle_interval))
        n_periods = len(opt.period)
        settle_idxs = [ x for x in range(opt.start_t, opt.start_t + n_periods) if x % opt.settle_interval == 0]
        opt.settle_period = [opt.period[i - opt.start_t] for i in settle_idxs]
    global DBG
    DBG = opt.dbg
    return 1

def remove_trailing_spaces(l):
    spaces = []
    m = []
    for x in l:
        if x == "":
            spaces.append(x)
        else:
            m.extend(spaces)
            m.append(x)
    return m
    
def read_script(script_csv, opt):
    """
    read script (csv file)
    """
    attrs = vars(opt)
    cfg = {}
    subsc_id_dict = {}   # subsc_id_id -> Subsc obj
    subsc_id_keys = []   # subsc_id_dict.keys() in the insertion order
    #n_vals = 0
    with open(script_csv, encoding="UTF-8") as fp:
        rp = csv.reader(fp)
        for i, row in enumerate(rp):
            [subsc_id, attr] = row[:2]
            if subsc_id == "":
                # global parameter; as if it is given in the command line
                vals = remove_trailing_spaces(row[2:])
                # vals = [v for v in vals if v != ""]
                if attr in attrs:
                    cfg[attr] = ",".join(vals) if len(vals) > 0 else None
                elif attr not in ["T", "C", "F0", "k0", "F1", "k1"]:
                    print(f"WARNING: global attribute {attr} in {script_csv} ignored (check typo)")
            else:
                sc = subsc_id_dict.get(subsc_id)
                if sc is None:
                    if DBG>=2:
                        print(f"make subsc {subsc_id}")
                    sc = {}
                    subsc_id_dict[subsc_id] = sc
                    subsc_id_keys.append(subsc_id)
                if attr in ["weight", "used"]:
                    val_cols = row[2:]
                    #n_vals = max(n_vals, len(val_cols))
                    vals = [parse_num(x, "") for x in val_cols]
                    if DBG>=2:
                        print(f'set Subsc("{subsc_id}").{attr} = {vals}')
                    sc[attr] = vals
    cfg["script"] = [(subsc_id, subsc_id_dict[subsc_id]) for subsc_id in subsc_id_keys]
    #return cfg, n_vals
    return cfg

class scenario_generator:
    """
    scenario generator
    """
    def __init__(self, opt):
        if opt.script is not None:
            #cfg, n_vals = read_script(opt.script, opt)
            cfg = read_script(opt.script, opt)
            opt_dict = vars(opt)
            for k, v in opt_dict.items():
                if v is None:
                    opt_dict[k] = cfg.get(k)
            script = cfg["script"]
            self.script = dict(script)
            # self.n_vals = n_vals
        else:
            self.script = None
            # self.n_vals = None
        self.ok = parse_opt_and_set_defaults(opt, self.script)
        # 失敗していることがある
        self.opt = opt
        self.rg = random.Random()
        self.rg.seed(opt.seed)
        self.subscs = set()

    def gen_new_subscs(self, t_idx, t):
        """
        t期の追加サブスク
        """
        new_subscs = []
        script = self.script
        if script is not None: # and t_idx < self.n_vals:
            for subsc_id, sb_script in script.items():
                used = sb_script["used"]
                if all(x == "" for x in used[:t_idx]) and t_idx < len(used) and used[t_idx] != "":
                    if DBG>=2:
                        print(f"t {t_idx} {t} gen_new_subscs : add subsc from script {subsc_id}")
                    assert(subsc_id not in self.subscs), (subsc_id, self.subscs)
                    self.subscs.add(subsc_id)
                    new_subscs.append(Subsc(subsc_id, t_idx))
        else:
            if t_idx == 0:
                delta_subscs = self.opt.init_subscs
            else:
                r = self.rg.random() * 2.0
                delta_subscs = int(self.opt.mean_delta_subscs * r)
            n_subscs = len(self.subscs)
            for i in range(n_subscs, n_subscs + delta_subscs):
                subsc_id = f"S{i:05}"
                if DBG>=2:
                    print(f"t {t_idx} {t} gen_new_subscs : add subsc {subsc_id}")
                assert(subsc_id not in self.subscs), (subsc_id, self.subscs)
                self.subscs.add(subsc_id)
                new_subscs.append(Subsc(subsc_id, t_idx))
        return new_subscs

    def gen_subsc_weight(self, t_idx, _t, sb):
        """
        t期のsubsc sbの重み
        """
        script = self.script
        if script is not None:
            sb_script = script[sb.subsc_id]
            weights = sb_script.get("weight")
            if weights is None:
                return None # no change
            elif t_idx < len(weights):
                return weights[t_idx]
            else:
                return None     # no change
        r = self.rg.random()
        if r < self.opt.weight_change_prob:
            # change
            s = self.rg.random()
            return 1.0 + 9.0 * s
        else:
            # no change
            return None

    def gen_use(self, t_idx, _t, F, sb, subscs):
        """
        t期のsubsc sbの使用量
        """
        script = self.script
        if script is not None:
            sb_script = script[sb.subsc_id]
            used = sb_script["used"]
            if t_idx < len(used):
                return used[t_idx]
            else:
                return 0.0
        W = sum(sb.weight for sb in subscs)
        algo = self.opt.user_algo
        if algo == USER_ALGO_C0:
            B = self.opt.C0 / self.opt.n_budget_periods
        elif algo == USER_ALGO_S:
            B = max(F, 0.0)
        else:
            assert(0), algo
        S = B * sb.weight / W
        if self.opt.random_use:
            # 平均 1
            r = gen_lognormal(self.rg, -0.5, 1.0)
        else:
            # 絶対 1
            r = 1.0
        return self.opt.mean_use_factor * S * r

def solve_q(a, b, c):
    """
    a x^2 + b x + c = 0 を解く
    """
    D = b * b - 4.0 * a * c
    assert(D >= 0.0), (a, b, c)
    k = (- b + math.sqrt(D)) / (2.0 * a)
    return k

class Simulator:
    """
    simulator
    """
    def __init__(self, opt):
        """
        simulator
        """
        self.sg = scenario_generator(opt)
        self.ok = self.sg.ok
        if not self.ok:
            return
        self.opt = opt
        self.subscs = []
        # クレジット残量
        C0 = opt.C0
        self.C = C0
        # timesteps
        # self.t = self.start_t
        # 目安消費量
        #  dT(t+1) = dT(t) + a         (dT(0) = b)
        #   T(t+1) =  T(t) + dT(t+1)   (T(0) = 0)
        # => dT(t) = at + b
        #     T(t) = T(0) + dT(t) + ... + dT(1)
        #          = a t(t+1) + bt
        # determine a and b such that
        #  0.5 a n (n + 1) = alpha * C0
        #      b n         =  beta * C0
        #  (alpha and beta are arbitrary numbers satisfying alpha + beta = 1)
        # =>
        nb = opt.n_budget_periods
        self.b = 0.4 * C0 / nb
        self.a = 0.6 * C0 / (0.5 * nb * (nb + 1))
        self.T = C0             # 目安残量
        self.hist_T = []        # 目安残量の履歴
        self.hist_C = []        # 実際の残量の履歴
        self.hist_F0 = []       # F0枠の履歴
        self.hist_k0 = []       # k0値の履歴
        self.hist_F1 = []       # F1枠の履歴
        self.hist_k1 = []       # k1値の履歴

    def check_subsc_lens(self, t_idx, _t):
        """
        length of history must match the period number
        """
        assert(len(self.hist_T) == t_idx), (t_idx, self.hist_T)
        assert(len(self.hist_C) == t_idx), (t_idx, self.hist_C)
        assert(len(self.hist_F0) == t_idx), (t_idx, self.hist_F0)
        assert(len(self.hist_k0) == t_idx), (t_idx, self.hist_k0)
        assert(len(self.hist_F1) == t_idx), (t_idx, self.hist_F1)
        assert(len(self.hist_k1) == t_idx), (t_idx, self.hist_k1)
        for sb in self.subscs:
            assert(len(sb.hist_weight) == t_idx), (t_idx, sb.hist_weight)
            assert(len(sb.hist_used) == t_idx), (t_idx, sb.hist_used)
            assert(len(sb.hist_free) == t_idx), (t_idx, sb.hist_free)
            assert(len(sb.hist_pend) == t_idx), (t_idx, sb.hist_pend)
            assert(len(sb.hist_disc) == t_idx), (t_idx, sb.hist_disc)
            assert(len(sb.hist_paid) == t_idx), (t_idx, sb.hist_paid)

    def sim_add_subscs(self, t_idx, t):
        """
        t期のサブスク追加
        """
        if DBG>=2:
            print(f"t {t_idx} {t} sim_add_subscs")
        new_subscs = self.sg.gen_new_subscs(t_idx, t)
        self.subscs.extend(new_subscs)

    def sim_change_weights(self, t_idx, t):
        """
        t期の重み変更
        """
        if DBG>=2:
            n_subscs = len(self.subscs)
            print(f"t {t_idx} {t} sim_change_weights : change weights of {n_subscs} subscs")
        for sb in self.subscs:
            weight = self.sg.gen_subsc_weight(t_idx, t, sb)
            if weight is None:
                if DBG>=3:
                    print(f"t {t_idx} {t} subsc={sb.subsc_id} : weight [{sb.weight:.2f}] did not change")
            else:
                if DBG>=2:
                    print(f"t {t_idx} {t} subsc {sb.subsc_id} : weight [{sb.weight:.2f}] -> {weight:.2f}")
                sb.weight = weight
            sb.hist_weight.append(sb.weight)

    def calc_F0(self, t_idx, t):
        """
        calc F0 value
        """
        T, C = self.T, self.C
        alpha = 0.6
        F0 = alpha * (C - T)
        if DBG>=2:
            print(f"t {t_idx} {t} calc_F0 : F0 = alpha [{alpha:.2f}] *"
                  f" (C [{C:.2f}] - T [{T:.2f}]) -> {F0:.2f}")
        assert(F0 >= 0.0), (C, T)
        self.hist_F0.append(F0)
        return F0

    def calc_F1(self, t_idx, t, do_settle):
        """
        calc F1 value
        """
        T, C = self.T, self.C
        if do_settle:
            F1 = C - T
        else:
            F1 = 0.0
        if DBG>=2:
            print(f"t {t_idx} {t} calc_F1 : do_settle = {do_settle} F1 = T [{T:.2f}] - C [{C:.2f}] -> {F1:.2f}")
        assert(F1 >= 0.0), (C, T)
        self.hist_F1.append(F1)
        return F1

    def record_S(self, t_idx, t, F0):
        """
        calc S value
        """
        W = sum(sb.weight for sb in self.subscs)
        if DBG>=2:
            print(f"t {t_idx} {t} record_S : F0 = {F0:.2f}, total weight W = {W:.2f}")
        for sb in self.subscs:
            sb.record_S(t_idx, t, F0, W)

    def sim_use_credit(self, t_idx, t, F0):
        """
        t期のクレジット利用 (F0値はF0)
        """
        if DBG>=2:
            print(f"t {t_idx} {t} sim_use_credit : F0 = {F0:.2f}")
        subscs = self.subscs
        for sb in subscs:
            u = self.sg.gen_use(t_idx, t, F0, sb, subscs)
            if DBG>=2:
                print(f"t {t_idx} {t} subsc={sb.subsc_id} : use {u:.2f}")
            assert(u >= 0.0), u
            assert(sb.used == 0.0), sb.used
            sb.used = u
            sb.hist_used.append(u)

    def distribute_F0_slow(self, _t_idx, _t, F0):
        """
        無料(F0)枠を配るための k を求める
        """
        subscs = [sb for sb in self.subscs if sb.weight > 0.0]
        sorted_subscs = sorted(subscs,
                               key=lambda sb: (sb.used + sb.pend) / sb.weight)
        if len(sorted_subscs) == 0:
            return 0.0
        for i, sb in enumerate(sorted_subscs):
            k = (sb.used + sb.pend) / sb.weight
            A = sum(sb.used + sb.pend for sb in sorted_subscs[:i])
            B = sum(sb.weight for sb in sorted_subscs[i:])
            if A + B * k >= F0:
                k = (F0 - A) / B
                assert(k >= 0.0), (F0, A, B)
                if F0 == 0.0:
                    assert(k == 0.0), (i, A, B)
                F0_ = sum(sb.F0(k) for sb in self.subscs)
                assert(very_close(F0_, F0)), (k, F0_, F0)
                return k
        F0_ = sum(sb.F0(k) for sb in self.subscs)
        assert(F0_ <= F0), (k, F0_, F0)
        return k

    def distribute_F0_fast(self, _t_idx, _t, F0):
        """
        無料(F0)枠を配るための k を求める
        """
        subscs = [sb for sb in self.subscs if sb.weight > 0.0]
        sorted_subscs = sorted(subscs,
                               key=lambda sb: (sb.used + sb.pend) / sb.weight)
        if len(sorted_subscs) == 0:
            return 0.0
        A = 0.0
        B = sum(sb.weight for sb in sorted_subscs)
        for i, sb in enumerate(sorted_subscs):
            k = (sb.used + sb.pend) / sb.weight
            # A = sum(sb.used + sb.pend for sb in sorted_subscs[:i])
            # B = sum(sb.weight for sb in sorted_subscs[i:])
            if A + B * k >= F0:
                k = (F0 - A) / B
                assert(k >= 0.0), (F0, A, B)
                if F0 == 0.0:
                    assert(k == 0.0), (i, A, B)
                F0_ = sum(sb.F0(k) for sb in self.subscs)
                assert(very_close(F0_, F0)), (k, F0_, F0)
                return k
            A += sb.used + sb.pend
            B -= sb.weight
        F0_ = sum(sb.F0(k) for sb in self.subscs)
        assert(F0_ <= F0), (k, F0_, F0)
        return k

    def distribute_F1_slow(self, _t_idx, _t, F1):
        """
        割引(F1)枠を配るための k を求める
        """
        subscs = [sb for sb in self.subscs if sb.weight > 0.0]
        sorted_subscs = sorted(subscs, key=lambda sb: sb.pend / sb.weight)
        if len(sorted_subscs) == 0:
            return 0.0
        for i, sb in enumerate(sorted_subscs):
            k = sb.pend / sb.weight
            A = sum(sb.weight for sb in sorted_subscs[i:]) / 2.0
            B = sum(sb.pend for sb in sorted_subscs[:i])
            C = sum(sb.pend ** 2 / sb.weight for sb in sorted_subscs[:i]) / 2.0
            if k > 0.0 and A * k ** 2 + B * k - C >= F1 * k:
                k = solve_q(A, B - F1, -C)
                assert(k >= 0.0), k
                if F1 == 0.0:
                    assert(k == 0.0), (i, A, B, C)
                F1_ = sum(sb.F1(k) for sb in self.subscs)
                assert(very_close(F1_, F1)), (k, F1_, F1)
                return k
        F1_ = sum(sb.F1(k) for sb in self.subscs)
        assert(F1_ <= F1), (k, F1_, F1)
        return k

    def distribute_F1_fast(self, _t_idx, _t, F1):
        """
        割引(F1)枠を配るための k を求める
        """
        subscs = [sb for sb in self.subscs if sb.weight > 0.0]
        sorted_subscs = sorted(subscs, key=lambda sb: sb.pend / sb.weight)
        if len(sorted_subscs) == 0:
            return 0.0
        A = sum(sb.weight for sb in sorted_subscs) / 2.0
        B = 0.0
        C = 0.0
        for i, sb in enumerate(sorted_subscs):
            k = sb.pend / sb.weight
            #A = sum(sb.weight for sb in sorted_subscs[i:]) / 2.0
            #B = sum(sb.pend for sb in sorted_subscs[:i])
            #C = sum(sb.pend ** 2 / sb.weight for sb in sorted_subscs[:i]) / 2.0
            if k > 0.0 and A * k ** 2 + B * k - C >= F1 * k:
                k = solve_q(A, B - F1, -C)
                assert(k >= 0.0), k
                if F1 == 0.0:
                    assert(k == 0.0), (i, A, B, C)
                F1_ = sum(sb.F1(k) for sb in self.subscs)
                assert(very_close(F1_, F1)), (k, F1_, F1)
                return k
            A -= sb.weight / 2.0
            B += sb.pend
            C += (sb.pend ** 2 / sb.weight) / 2.0
        F1_ = sum(sb.F1(k) for sb in self.subscs)
        assert(F1_ <= F1), (k, F1_, F1)
        return k

    def calc_at_period(self, t_idx, t, F0):
        """
        t 期の各サブスクの利用を見て
        どんだけ免除して, どんだけ未確定支払い予定コースに乗せて, 
        どんだけ支払い確定させるかを決める
        """
        # k値 (「k * 重み」以下は無料のライン) を計算
        if DBG>=2:
            print(f"t {t_idx} {t} calc_at_period : F0 = {F0:.2f}")
        k0 = self.distribute_F0_fast(t_idx, t, F0)
        if DBG>=2:
            print(f"t {t_idx} {t} calc_at_period : k0 -> {k0:.2f}")
        self.hist_k0.append(k0)
        # 各サブスク, 「最大k * 重みまでは無料」を反映
        # 今期使用量および未確定の超過クレからさっぴく
        W = 0.0
        for sb in self.subscs:
            W += sb.receive_free_credit(t_idx, t, k0)
        return W

    def calc_at_settle_period(self, t_idx, t, F1, do_settle):
        """
        (半)年度に一回とかそういうタイミングでの超過分の精算
        """
        if DBG>=2:
            print(f"t {t_idx} {t} calc_at_settle_period : F1 = {F1:.2f}, do_settle = {do_settle}")
        k1 = self.distribute_F1_fast(t_idx, t, F1)
        if DBG>=2:
            print(f"t {t_idx} {t} calc_at_settle_period : k1 -> {k1:.2f}")
        self.hist_k1.append(k1)
        W = 0.0
        for sb in self.subscs:
            W += sb.receive_discount_credit(t_idx, t, k1, do_settle)
        return W

    def update_C_at_period(self, t_idx, t, W0):
        """
        各期の残量更新
        """
        new_C = self.C - W0
        if DBG>=2:
            print(f"t {t_idx} {t} update_C_at_period : update C to reflect free credit"
                  f" C [{self.C:.2f}] -= W0 [{W0:.2f}] -> {new_C:.2f}")
        assert(new_C >= -1.0e-7), (self.C, W0)
        if new_C < 0.0:
            new_C = 0.0
        self.C = new_C

    def update_C_at_settle_period(self, t_idx, t, W1, do_settle):
        """
        各精算期の残量更新
        """
        new_C = self.C - W1
        if DBG>=2:
            print(f"t {t_idx} {t} update_C_at_period : do_settle [{do_settle}] update C to reflect discount"
                  f" C [{self.C:.2f}] -= W1 [{W1:.2f}] -> {new_C:.2f}")
        assert(new_C >= -1.0e-7), (self.C, W1)
        if new_C < 0.0:
            new_C = 0.0
        self.C = new_C
        self.hist_C.append(self.C)

    def update_target_consumption(self, t_idx, t):
        """
        目安残量(T)の更新
        """
        if DBG>=2:
            print(f"t {t_idx} {t} update_target_consumption :")
        new_b = self.b + self.a
        new_T = self.T - new_b
        if DBG>=2:
            print(f"t {t_idx} {t} update_target_consumption :"
                  f" dT [{self.b:.2f}] += a  [{self.a:.2f}] -> [{new_b:.2f}]")
            print(f"t {t_idx} {t} update_target_consumption :"
                  f"  T [{self.T:.2f}] -= dT [{new_b:.2f}] -> [{new_T:.2f}]")
        assert(new_T >= -1.0e-7), (self.b, new_b, self.T, new_T)
        if new_T < 0.0:
            new_T = 0.0
        self.b = new_b
        self.T = new_T
        self.hist_T.append(self.T)

    def sim_period(self, t_idx, t):
        """
        simulate a period
        """
        # 履歴の長さを確認
        self.check_subsc_lens(t_idx, t)
        # 目安消費量更新
        self.update_target_consumption(t_idx, t)
        # 新サブスクが加わる
        self.sim_add_subscs(t_idx, t)
        # 重みを変える
        self.sim_change_weights(t_idx, t)
        # F0枠計算
        F0 = self.calc_F0(t_idx, t)
        # S値記録
        self.record_S(t_idx, t, F0)
        # 使用
        self.sim_use_credit(t_idx, t, F0)
        # 各期の精算
        W0 = self.calc_at_period(t_idx, t, F0)
        self.update_C_at_period(t_idx, t, W0)
        # F1枠計算
        do_settle = (t in self.opt.settle_period)
        F1 = self.calc_F1(t_idx, t, do_settle)
        W1 = self.calc_at_settle_period(t_idx, t, F1, do_settle)
        self.update_C_at_settle_period(t_idx, t, W1, do_settle)

    def simulate(self):
        """
        n 期分シミュレート
        """
        for t_idx, t in enumerate(self.opt.period):
            self.sim_period(t_idx, t)

    def dump_history(self, hist_csv):
        """
        dump history
        """
        if DBG>=1:
            print(f"dump history to {hist_csv}")
        with open(hist_csv, "w", encoding="UTF-8") as wp:
            opt_dict = vars(self.opt)
            for k, v in opt_dict.items():
                if v is None:
                    dump_list(wp, ["", k], [])
                elif isinstance(v, type([])):
                    dump_list(wp, ["", k], v)
                else:
                    dump_list(wp, ["", k], [v])
            dump_list(wp, ["", "T"],  self.hist_T)
            dump_list(wp, ["", "C"],  self.hist_C)
            dump_list(wp, ["", "F0"],  self.hist_F0)
            dump_list(wp, ["", "k0"],  self.hist_k0)
            dump_list(wp, ["", "F1"],  self.hist_F1)
            dump_list(wp, ["", "k1"],  self.hist_k1)
            for sb in self.subscs:
                sb.dump_history(wp)

    def plot(self):
        """
        plot C and TC
        """
        if self.opt.plots is not None:
            import matplotlib.pyplot as plt
            n = min(len(self.opt.plots), len(self.subscs))
            a = int(math.sqrt(1 + n))
            b = (n + a) // a
            assert(a * b >= 1 + n), (n, a, b)
            _fig, axes = plt.subplots(a, b)
            if a * b == 1:
                axes = [axes]
            else:
                axes = axes.reshape(-1)
            # periods = list(range(self.opt.start_t, self.opt.end_t + 1))
            periods = list(range(len(self.opt.period)))
            ax = axes[0]
            ax.plot(periods, self.hist_T, label="target")
            ax.plot(periods, self.hist_C, label="actual")
            ax.set_ylim(bottom=0.0)
            ax.legend()
            ax.set_title("remaining credit")
            subscs_dict = {sb.subsc_id : sb for sb in self.subscs}
            for subsc_id, ax in zip(self.opt.plots, axes[1:]):
                sb = subscs_dict.get(subsc_id)
                if sb is None:
                    sb_idx = parse_num(subsc_id, None)
                    if isinstance(sb_idx, type(0)):
                        sb = self.subscs[sb_idx]
                if sb:
                    sb.plot_use_bars(ax, f"{sb.subsc_id}", periods)
                else:
                    print(f"WARNING: subsc {subsc_id} does not exist and cannot be plotted")
            plt.show()

def main():
    """
    main
    """
    opt = parse_args(sys.argv)
    sim = Simulator(opt)
    if not sim.ok:
        return
    sim.simulate()
    sim.dump_history(opt.hist)
    sim.plot()

main()
