import random

import torch

from nlprl import CharTokenizer, GPTConfig, SentimentTask, TinyGPT, get_task
from nlprl.algorithms import (DPOConfig, GRPOConfig, PPOConfig, RLConfig, train_dpo, train_grpo, train_ppo,
                              train_reinforce)
from nlprl.algorithms.dpo import dpo_loss
from nlprl.algorithms.grpo import group_advantages
from nlprl.algorithms.ppo import compute_gae
from nlprl.generation import generate, left_pad, token_logprobs
from nlprl.preference import make_preference_pairs, train_reward_model
from nlprl.pretrain import pretrain
from nlprl.utils import Logger

TOK = CharTokenizer()


def tiny_model(value_head=False):
    torch.manual_seed(0)
    return TinyGPT(GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_embd=32, n_head=2), with_value_head=value_head)


def test_tokenizer_roundtrip():
    s = "the movie was great."
    assert TOK.decode(TOK.encode(s, add_bos=True, add_eos=True)) == s


def test_left_padding_does_not_change_logprobs():
    """같은 시퀀스는 왼쪽 패딩 유무와 상관없이 같은 log-prob 을 가져야 한다."""
    m = tiny_model().eval()
    a = TOK.encode("the food was good.", add_bos=True)
    ids, mask = left_pad([a, TOK.encode("the movie was fun and nice.", add_bos=True)], TOK.pad_id)
    lp_padded = token_logprobs(m, ids[:1], mask[:1])
    lp_plain = token_logprobs(m, torch.tensor([a]), torch.ones(1, len(a), dtype=torch.long))
    n = len(a) - 1
    assert torch.allclose(lp_padded[0, -n:], lp_plain[0], atol=1e-5)


def test_generate_masks():
    m = tiny_model()
    roll = generate(m, TOK, ["the movie", "1+2="], max_new_tokens=8)
    assert roll.input_ids.shape == roll.response_mask.shape
    assert (roll.response_mask <= roll.attention_mask).all()
    assert roll.action_mask.shape[1] == roll.input_ids.shape[1] - 1


def test_rewards():
    t = SentimentTask()
    assert t.reward("the movie", " was great and fun.") == 2
    assert t.is_well_formed(" was really bad.")
    assert not t.is_well_formed(" was great great great.")
    a = get_task("arithmetic")
    assert a.reward("3+4=", "7.") == 1.0 and a.reward("3+4=", "8.") == 0.0


def test_gae_matches_monte_carlo_when_lambda_1():
    """γ=λ=1 이면 advantage = (남은 보상의 합) - V(s_t)."""
    rewards = torch.tensor([[0.0, 1.0, 0.0, 2.0]])
    values = torch.tensor([[0.5, 0.2, 0.1, 0.3]])
    mask = torch.tensor([[0.0, 1.0, 1.0, 1.0]])
    adv, ret = compute_gae(rewards, values * mask, mask, gamma=1.0, lam=1.0)
    assert torch.allclose(adv[0, 1:], torch.tensor([3.0, 2.0, 2.0]) - values[0, 1:])
    assert adv[0, 0] == 0


def test_group_advantages_zero_mean():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    a = group_advantages(r, 4)
    assert torch.allclose(a.view(2, 4).sum(1), torch.zeros(2), atol=1e-5)
    assert torch.all(a[4:] == 0)  # 모두 같은 보상이면 학습 신호 없음


def test_dpo_loss_at_init_is_log2():
    z = torch.zeros(4)
    loss, _, _ = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(loss.item() - torch.log(torch.tensor(2.0)).item()) < 1e-6


def _smoke(fn):
    rng = random.Random(0)
    torch.manual_seed(0)
    fn(rng)


def test_algorithms_smoke():
    task = get_task("sentiment")
    q = Logger(quiet=True)
    _smoke(lambda rng: train_reinforce(tiny_model(), TOK, task, RLConfig(steps=2, batch_size=8, eval_size=8), rng, logger=q))
    _smoke(lambda rng: train_ppo(tiny_model(True), TOK, task,
                                 PPOConfig(steps=2, batch_size=8, minibatch_size=4, eval_size=8), rng, logger=q))
    _smoke(lambda rng: train_grpo(tiny_model(), TOK, task,
                                  GRPOConfig(steps=2, batch_size=8, group_size=4, eval_size=8), rng, logger=q))


def test_preference_rm_dpo_smoke():
    rng = random.Random(0)
    task = get_task("sentiment")
    cfg = GPTConfig(vocab_size=TOK.vocab_size, n_layer=1, n_embd=32, n_head=2)
    policy = pretrain(task, TOK, rng, steps=150, cfg=cfg, log_every=1000, logger=Logger(quiet=True))
    pairs = make_preference_pairs(policy, TOK, task, rng, n_pairs=16)
    assert len(pairs) == 16
    train_reward_model(policy, TOK, pairs, rng, epochs=1, batch_size=8, logger=Logger(quiet=True))
    train_dpo(policy, TOK, task, pairs, DPOConfig(epochs=1, batch_size=8, log_every=1, eval_size=8), rng,
              logger=Logger(quiet=True))
