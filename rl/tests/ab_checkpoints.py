"""Head-to-head between two checkpoints.

    python3 rl/tests/ab_checkpoints.py [n_deals] [ckptA] [ckptB] [--deterministic] [--bo3]

Defaults to v7-victory8 vs v8-postfix, the pair it was written for.

`eval.duel` takes a heuristic AGENT as the opponent, which cannot be a network
(an agent is handed the state and the legal list, not an encoded observation),
so this drives the env directly and asks each net on its own turn. Seats are
swapped on every deal, because the first-player edge would otherwise decide it.
"""
import sys
import numpy as np, torch
from dataclasses import replace as dc_replace
from rl.engine.cardtable import full_table
from rl.config import Config
from rl.ppo import select_decks, deck_pool_deal
from rl.obs import Encoder
from rl.env import RiftboundEnv
from rl.eval import net_choice
from rl.engine.state import BoardOverflow

table = full_table()
cfg = dc_replace(Config().at_victory_score(8), units_only=False,
                 choose_payment="--old-payment" not in sys.argv)
pool = select_decks(table, None, 0.6)
deal = deck_pool_deal(table, 0.6, seed_decks=pool)

def load(p):
    ck = torch.load(p, map_location="cpu", weights_only=False)
    from rl.nets import build_net
    net = build_net(ck["shapes"], cfg) if callable(globals().get("build_net", None)) else None
    return ck

argv = [x for x in sys.argv[1:] if not x.startswith("--")]
DETERMINISTIC = "--deterministic" in sys.argv
# `--solver` gives side A the inference-time forced-win solver (BACKLOG #28).
# Pass the SAME checkpoint for both sides to measure the solver alone, with the
# policy held fixed -- then the only difference between the two players is the
# search, and nothing about net strength can confound it.
USE_SOLVER = "--solver" in sys.argv
# `--bo3` plays each deal as a whole Match (486) between the same two decks --
# sideboarding, battlefield picks and play order are the nets' own decisions --
# and scores MATCHES, with games reported alongside. Side A keeps its seat for
# the whole match; the seat swap is between deals, as in Bo1.
BO3 = "--bo3" in sys.argv
PATH_A = argv[1] if len(argv) > 1 else "rl/runs/v7-victory8/best.pt"
PATH_B = argv[2] if len(argv) > 2 else "rl/runs/v8-postfix/best.pt"
cka = torch.load(PATH_A, map_location="cpu", weights_only=False)
ckb = torch.load(PATH_B, map_location="cpu", weights_only=False)

# Rebuild both nets the way the Trainer does, then load the weights.
from rl.ppo import Trainer, HP
tr = Trainer(table, cfg, deal, HP(n_envs=2, rollout=16), device="cpu", seed=0)
import copy
# Grown into the current layout when they predate an append-only encoder
# change (Bo3) -- each plays exactly as it did; see `nets.load_weights`.
from rl.nets import load_weights
netA = copy.deepcopy(tr.net); load_weights(netA, cka, tr.vec.shapes()); netA.eval()
netB = copy.deepcopy(tr.net); load_weights(netB, ckb, tr.vec.shapes()); netB.eval()

enc = Encoder(table, cfg)
solver = None
if USE_SOLVER:
    from rl.solver import Solver
    solver = Solver()
    print("side A has the forced-win solver")
n = int(argv[0]) if argv else 300
seed0 = 300_000
gen = torch.Generator(device="cpu").manual_seed(seed0)
wins_a = wins_b = games = draws = 0
def play_game(env, obs, a_seat):
    """Play one game out from `obs`; the winner's seat, or -1."""
    try:
        steps = 0
        while obs is not None:
            steps += 1
            if steps > 5000:
                break
            mine = obs.to_move == a_seat
            net = netA if mine else netB
            idx = net_choice(net, obs, "cpu", DETERMINISTIC, gen)
            if solver is not None and mine:
                idx = solver.choice(env.state, table, cfg,
                                    int(obs.to_move), env.legal, idx)
            obs = env.step(idx).obs
        return int(env.state.winner)
    except BoardOverflow:
        return -1


if BO3:
    from rl.ppo import Bo3Dealer
    dealer = Bo3Dealer(table, pool, 0.6)
    m_a = m_b = m_un = g_a = g_b = g_un = 0
    lengths = [0, 0, 0]
    for i in range(n):
        for a_seat in (0, 1):
            env = RiftboundEnv(table, cfg, encoder=enc)
            slot = (i, a_seat)
            args, ctx = dealer.for_slot(slot, None, seed0 + i)
            while not ctx.over:
                obs = env.reset(seed0 + 10 * i + ctx.game, *args, match=ctx)
                w = play_game(env, obs, a_seat)
                ctx.record(env.state, table)
                if w < 0:
                    g_un += 1
                elif w == a_seat:
                    g_a += 1
                else:
                    g_b += 1
                if ctx.game > 6:                       # draws can repeat games
                    break
            lengths[min(ctx.game - 1, 3) - 1] += 1
            wa, wb = ctx.wins[a_seat], ctx.wins[1 - a_seat]
            if wa >= 2:
                m_a += 1
            elif wb >= 2:
                m_b += 1
            else:
                m_un += 1
        if (i + 1) % 25 == 0:
            print(f"  {2 * (i + 1)} matches: A {m_a}  B {m_b}  undecided {m_un}"
                  f"  | games A {g_a}  B {g_b}  undecided {g_un}", flush=True)
    for label, wa, wb, un in (("matches", m_a, m_b, m_un), ("games", g_a, g_b, g_un)):
        dec = wa + wb
        p = wa / max(1, dec)
        ci = 1.96 * (p * (1 - p) / max(1, dec)) ** 0.5
        print(f"\n{label}: {dec + un} ({dec} decided, {un} undecided)")
        print(f"A  {PATH_A} (iter {cka.get('iter')}):  {wa}  = {p:.1%} +/- {ci:.1%}")
        print(f"B  {PATH_B} (iter {ckb.get('iter')}):  {wb}  = {1-p:.1%}")
    print(f"match lengths (1/2/3+ games): {lengths}")
    print(f"sampled actions" if not DETERMINISTIC else "deterministic (argmax)")
    sys.exit(0)

for i in range(n):
    for a_seat in (0, 1):
        env = RiftboundEnv(table, cfg, encoder=enc)
        obs = env.reset(seed0 + i, *deal(seed0 + i))
        try:
            steps = 0
            while obs is not None:
                steps += 1
                if steps > 5000:
                    break
                mine = obs.to_move == a_seat
                net = netA if mine else netB
                idx = net_choice(net, obs, "cpu", DETERMINISTIC, gen)
                if solver is not None and mine:
                    idx = solver.choice(env.state, table, cfg,
                                        int(obs.to_move), env.legal, idx)
                obs = env.step(idx).obs
            w = int(env.state.winner)
        except BoardOverflow:
            w = -1
        games += 1
        if w < 0:
            draws += 1
        elif w == a_seat:
            wins_a += 1
        else:
            wins_b += 1
    if (i + 1) % 50 == 0:
        print(f"  {games} games: A {wins_a}  B {wins_b}  undecided {draws}",
              flush=True)

dec = wins_a + wins_b
p = wins_a / max(1, dec)
ci = 1.96 * (p * (1 - p) / max(1, dec)) ** 0.5
print(f"\n{games} games ({dec} decided, {draws} undecided)")
print(f"A  {PATH_A} (iter {cka.get('iter')}):  {wins_a}  = {p:.1%} +/- {ci:.1%}")
print(f"B  {PATH_B} (iter {ckb.get('iter')}):  {wins_b}  = {1-p:.1%}")
print(f"sampled actions" if not DETERMINISTIC else "deterministic (argmax)")
if solver is not None:
    print(solver.summary())
