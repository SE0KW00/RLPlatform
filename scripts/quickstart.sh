#!/usr/bin/env bash
# 전체 커리큘럼의 실험을 한 번에 실행한다 (CPU 약 10분).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m nlprl pretrain --task sentiment
python -m nlprl train --task sentiment --algo reinforce
python -m nlprl train --task sentiment --algo ppo
python -m nlprl train --task sentiment --algo ppo --kl-coef 0.05 --name ppo_hack
python -m nlprl reward-model --task sentiment
python -m nlprl train --task sentiment --algo ppo --reward rm
python -m nlprl dpo --task sentiment

python -m nlprl pretrain --task arithmetic
python -m nlprl train --task arithmetic --algo grpo
python -m nlprl train --task arithmetic --algo ppo

python -m nlprl plot runs/sentiment/*/history.json --metric eval/positive_rate --out runs/sentiment.png
python -m nlprl plot runs/sentiment/*/history.json --metric eval/well_formed --out runs/sentiment_well_formed.png
python -m nlprl plot runs/arithmetic/*/history.json --metric eval/accuracy --out runs/arithmetic.png
echo "그래프: runs/*.png"
