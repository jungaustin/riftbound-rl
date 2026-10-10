"""Network -- PLAN.md §6.

Three pieces, and the middle one is the whole reason this architecture was
chosen over anything standard.

**Trunk (§6.1).** Every zone is a *set* of cards, so each is encoded by the same
shared per-card MLP and pooled permutation-invariantly (DeepSets: masked mean
concatenated with masked max). One shared encoder across all four zones works
because each row carries its own zone one-hot in the context block -- so "a
2-might unit in my hand" and "a 2-might unit on a battlefield" are the same
function applied to different inputs, which is exactly the weight sharing you
want.

**Candidate-scoring action head (§6.2).** The net does not have an output neuron
per action. It scores each *featurized* legal action against the trunk state and
softmaxes over the scores. Two consequences, both load-bearing:

  - Variable action counts are native. No padding semantics to get wrong beyond
    the mask, and no fixed action vocabulary to outgrow when spells land.
  - An action involving a card the net never saw in training still gets a
    sensible score, because the action row carries the card's *attributes*, not
    its id. That is what makes the deck-evaluator framing (§1.1) reachable at
    all -- a policy indexed by card id could only ever play the decks it trained
    on.

**Asymmetric value head (§6.4).** The critic additionally sees the opponent's
hand and every facedown card. It is discarded at play time, so the policy stays
honest; during training it removes a large chunk of the variance that comes from
the critic having to guess what it is being punished for. The privileged input
is a *set* of cards too, so it gets the same DeepSets treatment.

**Entropy is normalized (§6.3).** Softmax entropy is bounded by log(n_legal),
and n_legal swings from 2 to 10+ here. A fixed coefficient would pay the policy
to steer toward high-branching states for reasons unrelated to winning, which is
a bias that looks exactly like strategy.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = -1e9   # not -inf: a fully-masked row would produce NaN in softmax


def mlp(sizes: list[int], out_act: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if out_act or i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


def init_(module: nn.Module, gain: float = math.sqrt(2), bias: float = 0.0):
    """Orthogonal init. The policy head gets gain 0.01 so the initial policy is
    near-uniform over legal actions -- a confidently wrong start is much harder
    for PPO to walk back than an indifferent one."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain)
            if m.bias is not None:      # the policy scorer is bias-free
                nn.init.constant_(m.bias, bias)
    return module


def masked_pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """[B, N, D] + [B, N] -> [B, 2D]: masked mean concatenated with masked max.

    Mean alone loses "is there a 5-might unit here"; max alone loses "how many".
    Both matter in this game and the pair is cheap.
    """
    m = mask.unsqueeze(-1).to(x.dtype)
    n = m.sum(1).clamp(min=1.0)
    mean = (x * m).sum(1) / n
    mx = x.masked_fill(~mask.unsqueeze(-1), NEG_INF).max(1).values
    # A zone with nothing in it pools to zeros rather than to -1e9.
    empty = ~mask.any(1, keepdim=True)
    return torch.cat([mean, torch.where(empty, torch.zeros_like(mx), mx)], -1)


class RiftboundNet(nn.Module):
    def __init__(self, shapes: dict, hidden: int = 256, card_hidden: int = 128,
                 head_hidden: int = 128,
                 privileged_critic: bool = False) -> None:
        super().__init__()
        self.privileged_critic = bool(privileged_critic)
        # From the encoder, never a copy -- see `obs.ZONES`. The copy that
        # used to live here missed `legends` for every run after a8a7bf4.
        self.zone_names = tuple(shapes["zones"])
        row_dim = shapes["row_dim"]
        card_dim = shapes["card_dim"]
        g_dim = shapes["globals"][0]
        act_dim = shapes["actions"][1]
        priv_dim = shapes["privileged"][0]

        self.card_dim = card_dim
        self.priv_slots = priv_dim // card_dim

        self.card_enc = init_(mlp([row_dim, card_hidden, card_hidden], True))
        trunk_in = 2 * card_hidden * len(self.zone_names) + g_dim
        self.trunk = init_(mlp([trunk_in, hidden, hidden], True))

        # Policy: score each candidate action against the trunk state.
        #
        # The last layer has **no bias**, and that is not a tuning choice: it
        # emits one score per candidate and the scores go through a softmax,
        # which is invariant to adding the same constant to all of them. A bias
        # there is added identically to every candidate and cancels exactly, so
        # its gradient is structurally zero. Leaving it in made `test_ppo`'s
        # "every parameter moved" check pass on floating-point noise and fail at
        # random -- a flaky test pointing at a parameter that could never move.
        self.act_head = init_(mlp([hidden + act_dim, head_hidden, head_hidden, 1]))
        self.act_head[-1] = nn.Linear(head_hidden, 1, bias=False)
        init_(self.act_head[-1], gain=0.01)

        # Two critics, and which one drives learning is a knob (§6.4).
        #
        # `value_sym` sees exactly what the policy sees. `value_priv` also sees
        # the opponent's hand and every facedown card.
        #
        # **Why the privileged one is no longer the default.** As a pure
        # baseline it is free: the policy gradient is
        # `grad log pi(a|o) * A`, and a baseline that depends on hidden state is
        # still independent of the ACTION given the state, so its expected
        # contribution is exactly zero. But GAE does not use V only as a
        # baseline -- `V(s_{t+1})` appears inside the TD residual, and there it
        # carries information the policy does not have. The distortion shrinks
        # as gae_lambda -> 1 (0.95 here, so it is small) but it is not zero, and
        # the direction it errs in is the bad one: a reckless line that the
        # opponent happened not to hold the answer to gets less blame than it
        # earned, because the critic knew they could not punish it.
        #
        # Both heads are always trained, so `explained_variance` can be read
        # for each and the choice settled by measurement rather than by
        # argument -- and so an honest, policy-side win probability is always
        # available for replaying a position back to a human.
        self.priv_enc = init_(mlp([card_dim, card_hidden, card_hidden], True))
        self.value_sym = init_(mlp([hidden, hidden, 1]))
        init_(self.value_sym[-1], gain=1.0)
        self.value_priv = init_(mlp([hidden + 2 * card_hidden, hidden, 1]))
        init_(self.value_priv[-1], gain=1.0)

    # -- forward ---------------------------------------------------------

    def encode(self, zones, zone_mask, globals_) -> torch.Tensor:
        pooled = [masked_pool(self.card_enc(zones[z]), zone_mask[z])
                  for z in self.zone_names]
        return self.trunk(torch.cat(pooled + [globals_], -1))

    def logits(self, h: torch.Tensor, actions: torch.Tensor,
               action_mask: torch.Tensor) -> torch.Tensor:
        b, a, _ = actions.shape
        x = torch.cat([h.unsqueeze(1).expand(b, a, h.shape[-1]), actions], -1)
        return self.act_head(x).squeeze(-1).masked_fill(~action_mask, NEG_INF)

    def values(self, h: torch.Tensor,
               privileged: torch.Tensor | None) -> tuple:
        """`(value used for learning, the other one)`, per `privileged_critic`.

        **The auxiliary head is fed a DETACHED trunk.** Both critics share the
        trunk with the policy, so whichever one is not driving learning must not
        shape that representation -- otherwise turning the privileged critic
        "off" would still let hidden-information gradients steer the features
        the policy reads, which is the thing switching it off is for. Detached,
        the spare head is a pure measurement.
        """
        if privileged is None:
            z = torch.zeros(h.shape[0], 2 * self.priv_enc[-2].out_features,
                            device=h.device, dtype=h.dtype)
        else:
            cards = privileged.view(-1, self.priv_slots, self.card_dim)
            z = masked_pool(self.priv_enc(cards), (cards != 0).any(-1))
        if self.privileged_critic:
            main = self.value_priv(torch.cat([h, z], -1)).squeeze(-1)
            aux = self.value_sym(h.detach()).squeeze(-1)
        else:
            main = self.value_sym(h).squeeze(-1)
            aux = self.value_priv(torch.cat([h.detach(), z], -1)).squeeze(-1)
        return main, aux

    def win_prob(self, obs: dict) -> torch.Tensor:
        """The POLICY-side value, always, whatever drives learning.

        This is the number to show a person: with terminal-only reward and
        gamma=1 the critic is a win probability, and it is only honest as a
        teaching signal if it was computed from what a player could actually
        see. `value_priv` would answer "given the opponent's hand", which is
        not a question anyone sitting at the table can ask.
        """
        h = self.encode(obs["zones"], obs["zone_mask"], obs["globals"])
        return self.value_sym(h).squeeze(-1)

    def forward(self, obs: dict):
        h = self.encode(obs["zones"], obs["zone_mask"], obs["globals"])
        main, aux = self.values(h, obs.get("privileged"))
        return (self.logits(h, obs["actions"], obs["action_mask"]), main, aux)

    # -- sampling --------------------------------------------------------

    def act(self, obs: dict, deterministic: bool = False,
            generator: torch.Generator | None = None):
        """Returns (action_index, logprob, entropy, value), all [B]."""
        logits, value, _aux = self(obs)
        logp = F.log_softmax(logits, -1)
        if deterministic:
            idx = logits.argmax(-1)
        else:
            # `torch.multinomial` on the exponentiated log-probs rather than
            # Categorical(...).sample(), so the draw can take an explicit
            # generator and evaluation stays reproducible.
            idx = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(-1)
        return idx, logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1), \
            self.entropy(logits, obs["action_mask"]), value

    @staticmethod
    def entropy(logits: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Entropy normalized by log(n_legal) -- §6.3.

        Unnormalized, the bonus is worth more in states with more legal actions,
        so the policy learns to keep its options open for reasons that have
        nothing to do with winning.
        """
        logp = F.log_softmax(logits, -1)
        p = logp.exp()
        ent = -(p * logp.masked_fill(~action_mask, 0.0)).sum(-1)
        n = action_mask.sum(-1).clamp(min=2).to(ent.dtype)
        return ent / n.log()

    def evaluate(self, obs: dict, idx: torch.Tensor):
        """Log-prob, entropy and both values of `idx` under the current params.

        The fourth return is the critic that is NOT driving learning; it is
        fitted to the same returns so the two can be compared, and its gradient
        cannot reach the trunk. See `values`.
        """
        logits, value, aux = self(obs)
        logp = F.log_softmax(logits, -1)
        return (logp.gather(-1, idx.unsqueeze(-1)).squeeze(-1),
                self.entropy(logits, obs["action_mask"]), value, aux)


# ---------------------------------------------------------------------------
# numpy <-> torch
# ---------------------------------------------------------------------------

def to_torch(b, device: str = "cpu", privileged: bool = True) -> dict:
    """A `BatchObs` (or a dict of numpy arrays) as model input."""
    t = lambda x, d=torch.float32: torch.as_tensor(x, dtype=d, device=device)
    out = {
        "zones": {k: t(v) for k, v in b.zones.items()},
        "zone_mask": {k: t(v, torch.bool) for k, v in b.zone_mask.items()},
        "globals": t(b.globals),
        "actions": t(b.legal_actions),
        "action_mask": t(b.action_mask, torch.bool),
    }
    if privileged and b.privileged is not None:
        out["privileged"] = t(b.privileged)
    return out


def count_params(net: nn.Module) -> int:
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Growing a checkpoint into a wider observation layout.
#
# An observation change used to mean "retrain from scratch": every input layer's
# width is baked into the checkpoint. But when the change only ADDS inputs -- a
# zone, some globals, row columns, action kinds -- the old network can be copied
# into the new shapes exactly, with every new input column zero-weighted, and it
# then computes the SAME function it always did on any position where the new
# inputs say nothing. That is what lets a Bo3 run warm-start from v14b and learn
# the new decisions on top of everything v14b already knows.
#
# The rule that makes it safe: new things are only ever APPENDED -- new row
# columns after the old ones, new zones after the old zones, new globals after
# the old globals, new action kinds after the old kinds. The one non-tail
# insertion is the action kinds, which sit before the card block in the action
# row; `act_split` says where.

PRE_BO3_ACT_SPLIT = (25, 20)    # (action kinds, ACT_EXTRA) before D2/D2b
BO3_ACT_SPLIT = (31, 22)        # ...from D2/D2b until the payment choice
PAY_ACT_SPLIT = (32, 28)        # ...from D6 until the mine / in-combat columns
# Every action-row layout a checkpoint may have been saved with, oldest first.
# A checkpoint records only the row's total width, and the splits differ in
# total, so the width alone says which one it is.
KNOWN_ACT_SPLITS = (PRE_BO3_ACT_SPLIT, BO3_ACT_SPLIT, PAY_ACT_SPLIT)


def untrained_choice(net, obs) -> int | None:
    """Option 0 (the engine's old default) when `net` predates the decision on
    offer, else None. Only for PLAYING a checkpoint -- training must explore."""
    from rl.engine import actions as _A
    if getattr(net, "knows_payment", True) or obs.n_legal == 0:
        return None
    return 0 if obs.legal_actions[0, _A.A_PAY_SPLIT] == 1.0 else None


def act_split_of(shapes: dict) -> tuple[int, int]:
    """The (kinds, ACT_EXTRA) a checkpoint with these shapes was saved with."""
    from rl.engine import actions as _A
    from rl.obs import ACT_EXTRA as _EXTRA
    width = shapes["actions"][1] - 1 - shapes["card_dim"]
    for split in KNOWN_ACT_SPLITS + ((len(_A.KIND_NAMES), _EXTRA),):
        if sum(split) == width:
            return split
    raise ValueError(f"no known action-row layout is {width} wide")


def shapes_growable(old: dict, new: dict) -> bool:
    """Can a checkpoint with layout `old` be grown into `new`?"""
    if old == new:
        return True
    return (old["card_dim"] == new["card_dim"]
            and old["row_dim"] <= new["row_dim"]
            and tuple(new["zones"][:len(old["zones"])]) == tuple(old["zones"])
            and old["globals"][0] <= new["globals"][0]
            and old["actions"][1] <= new["actions"][1]
            and old["privileged"] == new["privileged"])


def grow_state_dict(sd: dict, old: dict, new: dict,
                    act_split_old: tuple[int, int] = PRE_BO3_ACT_SPLIT,
                    act_split_new: tuple[int, int] | None = None) -> dict:
    """`sd` (trained on layout `old`) re-laid-out for `new`, zeros for the new.

    Exactly three layers read the observation -- `card_enc.0` (row columns),
    `trunk.0` (pooled zones then globals) and `act_head.0` (trunk state then the
    action row) -- and only those change width. Everything else is copied.
    """
    assert shapes_growable(old, new), "layouts are not growable"
    if old == new:
        return dict(sd)
    from rl.engine import actions as _A
    from rl.obs import ACT_EXTRA as _EXTRA
    nk_new, ex_new = act_split_new or (len(_A.KIND_NAMES), _EXTRA)
    nk_old, ex_old = act_split_old
    card_dim = old["card_dim"]
    assert nk_old + 1 + card_dim + ex_old == old["actions"][1], \
        "act_split_old does not describe this checkpoint's action row"
    assert nk_new + 1 + card_dim + ex_new == new["actions"][1], \
        "act_split_new does not describe the current action row"
    out = dict(sd)

    w = sd["card_enc.0.weight"]
    g = torch.zeros(w.shape[0], new["row_dim"], dtype=w.dtype)
    g[:, :old["row_dim"]] = w
    out["card_enc.0.weight"] = g

    w = sd["trunk.0.weight"]
    ch2 = 2 * sd["card_enc.0.weight"].shape[0]      # mean + max per zone
    nz_old, nz_new = len(old["zones"]), len(new["zones"])
    g = torch.zeros(w.shape[0], ch2 * nz_new + new["globals"][0], dtype=w.dtype)
    for k, z in enumerate(old["zones"]):
        j = list(new["zones"]).index(z)
        g[:, j * ch2:(j + 1) * ch2] = w[:, k * ch2:(k + 1) * ch2]
    g[:, ch2 * nz_new:ch2 * nz_new + old["globals"][0]] = w[:, ch2 * nz_old:]
    out["trunk.0.weight"] = g

    w = sd["act_head.0.weight"]
    hidden = sd["trunk.0.weight"].shape[0]
    tail = 1 + card_dim + ex_old
    g = torch.zeros(w.shape[0], hidden + new["actions"][1], dtype=w.dtype)
    g[:, :hidden + nk_old] = w[:, :hidden + nk_old]
    g[:, hidden + nk_new:hidden + nk_new + tail] = w[:, hidden + nk_old:]
    out["act_head.0.weight"] = g
    return out


def load_weights(net: nn.Module, ck: dict, shapes: dict) -> bool:
    """Load checkpoint `ck` into `net` (built for `shapes`), growing if needed.

    Returns True when the checkpoint was grown, so a caller can say so.
    """
    old = ck["shapes"]
    # A checkpoint from before the payment choice has never seen `A_PAY_SPLIT`:
    # every option encodes identically to it (the split columns are
    # zero-weighted), so it would pick uniformly. `untrained_choice` lets an
    # evaluator give it the engine's old default -- option 0 -- instead.
    net.knows_payment = act_split_of(old) not in (PRE_BO3_ACT_SPLIT,
                                                  BO3_ACT_SPLIT)
    if old == shapes:
        net.load_state_dict(ck["net"])
        return False
    if not shapes_growable(old, shapes):
        raise ValueError(f"checkpoint layout cannot be grown into the current "
                         f"one:\n  ckpt {old}\n  here {shapes}")
    net.load_state_dict(grow_state_dict(ck["net"], old, shapes,
                                        act_split_old=act_split_of(old)))
    return True
