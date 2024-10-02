# python train_text_cls_ori_v2.py --order 1 && python train_text_cls_ori_v2.py --order 2 && python train_text_cls_ori_v2.py --order 3 && python train_text_cls_ori_v2.py --order 4


# Train with Inner LR 2x
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.002

# Test with Inner LR 5x
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005
# python train_text_cls_ori_v2.py --order 1 --updates 20 --inner_lr 0.005

# Test with Inner LR 10x
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01

# Test with Inner LR 20x
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.02

# Test with Inner LR 50x
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.05


# Train with Inner LR 10x, SR
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01 --selective_replay
# Train with Inner LR 10x, SR All Query
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01 --selective_replay --all_query_replay


# Train with Inner LR 10x, SR All Query + Key-based Validation
# 575k * 0.1 * 0.2 ~ 11.5k for validation set.
# python train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01 --selective_replay --all_query_replay --validation_split 0.2 --write_prob 0.1


# Train with Inner LR 10x, SR, Task Aware, Reverse Support , write_prob 0.1
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01 --selective_replay --task_aware --reverse_support --write_prob 0.1

# python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01 --selective_replay --task_aware --reverse_support --write_prob 0.1

# Train with Inner LR 10x, SR, Task Aware, Reverse Support , write_prob 0.1, replay_steps = 5
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000

# Version 11: with curriculum replay
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay

## Check if loss.backward() works!!
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model bert --pln 1fc


# Version 12: with roberta base, and fc 2 layers
# The final version to use is fc 1 layer..
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc

## This is final version>>> 
python3.8 train_text_cls_ori_v2.py --order 4 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc

python3.8 train_text_cls_ori_v2.py --order 3 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc && python3.8 train_text_cls_ori_v2.py --order 2 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc


# Version 13: Adapter
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc --adapter

python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 5e-3 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc --adapter --inner_adapter_lr 5e-3 --adapter_type pfeiffer --mini_batch_size 12


# SEQUENTIAL
python3.8 train_text_cls_ori_v2.py --order 1 --learner sequential --model roberta


# Version Ablation #1. No Adapt Score (Change code instead @ 4 points)
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc
python3.8 train_text_cls_ori_v2.py --order 2 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc  && \
python3.8 train_text_cls_ori_v2.py --order 3 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc  && \
python3.8 train_text_cls_ori_v2.py --order 4 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --curriculum_replay --model roberta --pln 1fc

# Version Ablation #2. No Selective Replay 
# Order 2 Start run @ 2023-09-26 5:54PM  Should End at 7*3~21 Hours later @2023-09-27 2:54PM
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --reverse_support --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc
python3.8 train_text_cls_ori_v2.py --order 3 --updates 5 --inner_lr 0.005 --reverse_support --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc && \
python3.8 train_text_cls_ori_v2.py --order 4 --updates 5 --inner_lr 0.005 --reverse_support --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc

# Version Ablation #3. No Curriculum Replay
# Order 2 Start run @ 2023-09-28 12:02AM  Should End at 8*3~24 Hours later @2023-09-29 12:02AM
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc
python3.8 train_text_cls_ori_v2.py --order 2 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc  && \
python3.8 train_text_cls_ori_v2.py --order 3 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc  && \
python3.8 train_text_cls_ori_v2.py --order 4 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc


# Version Ablation #4. Train w/ DS Split like OML
# Order 2 Start run @ 2023-09-29 12:02AM  Should End at 8*3~24 Hours later @2023-09-30 12:02AM
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc
python3.8 train_text_cls_ori_v2.py --order 2 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc && \
python3.8 train_text_cls_ori_v2.py --order 3 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc && \
python3.8 train_text_cls_ori_v2.py --order 4 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --replay_every 8000 --model roberta --pln 1fc

# Version Ablation #5. Train w/ All Query like MAML-CL
# Order 2 Start run @ 2023-10-02 2:02AM  Should End at 8*3~24 Hours later @2023-10-03 2:02AM
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --model roberta --pln 1fc --all_query_replay
python3.8 train_text_cls_ori_v2.py --order 2 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --model roberta --pln 1fc --all_query_replay && \
python3.8 train_text_cls_ori_v2.py --order 3 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --model roberta --pln 1fc --all_query_replay && \
python3.8 train_text_cls_ori_v2.py --order 4 --updates 5 --inner_lr 0.005 --selective_replay --task_aware --write_prob 0.1 --model roberta --pln 1fc --all_query_replay