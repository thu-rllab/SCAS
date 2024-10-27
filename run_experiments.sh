python main.py --env halfcheetah-medium-expert-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env hopper-medium-expert-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env walker2d-medium-expert-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env halfcheetah-medium-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env hopper-medium-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env walker2d-medium-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env halfcheetah-medium-replay-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env hopper-medium-replay-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env walker2d-medium-replay-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env halfcheetah-random-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env hopper-random-v2 --temp 5.0 --lam 0.25 --beta 3e-3
python main.py --env walker2d-random-v2 --temp 5.0 --lam 0.25 --beta 3e-3

python main.py --env antmaze-umaze-v2 --temp 5.0 --lam 0.25 --beta 3e-3 --eval_episodes 100 --eval_freq 50000
python main.py --env antmaze-umaze-diverse-v2 --temp 5.0 --lam 0.25 --beta 3e-3 --eval_episodes 100 --eval_freq 50000
python main.py --env antmaze-medium-play-v2 --temp 5.0 --lam 0.25 --beta 3e-3 --eval_episodes 100 --eval_freq 50000
python main.py --env antmaze-medium-diverse-v2 --temp 5.0 --lam 0.25 --beta 3e-3 --eval_episodes 100 --eval_freq 50000
python main.py --env antmaze-large-play-v2 --temp 5.0 --lam 0.25 --beta 3e-3 --eval_episodes 100 --eval_freq 50000 --no_normalize
python main.py --env antmaze-large-diverse-v2 --temp 5.0 --lam 0.25 --beta 3e-3 --eval_episodes 100 --eval_freq 50000 --no_normalize