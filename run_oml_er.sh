# python3.8 train_text_cls_ori_v2.py --order 4 --updates 5 --inner_lr 0.01 --write_prob 0.01 --model roberta

# -----------------For ORDER 4------------------
# 2023-04-25 07:04:30,957 - OML-Log - INFO - Loaded TransformerRLN as RLN                                
# 2023-04-25 07:04:30,957 - OML-Log - INFO - Loaded LinearPLN as PLN                                     
# 2023-04-25 07:04:30,960 - ContinualLearningLog - INFO - Using OML as learner                           
# 2023-04-25 07:04:30,960 - ContinualLearningLog - INFO - ----------Training starts here----------       
# 2023-04-25 07:04:30,960 - OML-Log - INFO - Replay frequency: 101                                       
# 2023-04-25 07:04:30,960 - OML-Log - INFO - Replay steps: 6                                             
# 2023-04-25 07:04:30,960 - OML-Log - INFO - Starting with train_idx: 0                                  
# <class 'torch.utils.data.dataset.Subset'>                                                              
# 100%|██████████████████████████████████████████████████████████▉| 7184/7188 [25:03<00:00,  4.50it/s]
# 2023-04-25 07:29:35,120 - OML-Log - INFO - Terminating training as all the data is seen                   
# 100%|███████████████████████████████████████████████████████████| 7188/7188 [25:04<00:00,  4.78it/s]   
# 2023-04-25 07:29:35,121 - OML-Log - INFO - Saving Model with train_idx: 0                              
# 2023-04-25 07:29:35,944 - OML-Log - INFO - Starting with train_idx: 1                                  
# <class 'torch.utils.data.dataset.Subset'>                                                              
# 100%|███████████████████████████████████████████████████████████| 7188/7188 [26:30<00:00,  4.94it/s]2023-04-25 07:56:07,015 - OML-Log - INFO - Terminating training as all the data is seen
# 2023-04-25 08:21:49,378 - OML-Log - INFO - Saving Model with train_idx: 2
# 2023-04-25 08:21:50,242 - OML-Log - INFO - Starting with train_idx: 3
# <class 'torch.utils.data.dataset.Subset'>
# 100%|███████████████████████████████████████████████████████████| 7188/7188 [26:22<00:00,  4.99it/s]2023-04-25 08:48:12,575 - OML-Log - INFO - Terminating training as all the data is seen
# /root/MetaLifelongLanguage/env/lib/python3.8/site-packages/numpy/lib/function_base.py:518: RuntimeWarning: Mean of empty slice.
#   avg = a.mean(axis, **keepdims_kw)
# /root/MetaLifelongLanguage/env/lib/python3.8/site-packages/numpy/core/_methods.py:190: RuntimeWarning: invalid value encountered in double_scalars
#   ret = ret.dtype.type(ret / rcount)
# 100%|███████████████████████████████████████████████████████████| 7188/7188 [26:22<00:00,  4.54it/s]   2023-04-25 08:48:12,579 - OML-Log - INFO - Saving Model with train_idx: 3
# 2023-04-25 08:48:13,424 - OML-Log - INFO - Starting with train_idx: 4
# <class 'torch.utils.data.dataset.Subset'>
# 100%|███████████████████████████████████████████████████████████| 7188/7188 [25:25<00:00,  5.15it/s]2023-04-25 09:13:38,951 - OML-Log - INFO - Terminating training as all the data is seen
# /root/MetaLifelongLanguage/env/lib/python3.8/site-packages/numpy/lib/function_base.py:518: RuntimeWarning: Mean of empty slice.
#   avg = a.mean(axis, **keepdims_kw)
# /root/MetaLifelongLanguage/env/lib/python3.8/site-packages/numpy/core/_methods.py:190: RuntimeWarning: invalid value encountered in double_scalars
#   ret = ret.dtype.type(ret / rcount)
# 100%|███████████████████████████████████████████████████████████| 7188/7188 [25:25<00:00,  4.71it/s]   2023-04-25 09:13:38,954 - OML-Log - INFO - Saving Model with train_idx: 4
# 2023-04-25 09:13:39,780 - ContinualLearningLog - INFO - [TIME] End Run at 2023-04-25T09:13:39 within 2.1565424559513726 hours


python3.8 train_text_cls_ori_v2.py --order 3 --updates 5 --inner_lr 0.01 --write_prob 0.01 --model roberta
python3.8 train_text_cls_ori_v2.py --order 2 --updates 5 --inner_lr 0.01 --write_prob 0.01 --model roberta
python3.8 train_text_cls_ori_v2.py --order 1 --updates 5 --inner_lr 0.01 --write_prob 0.01 --model roberta