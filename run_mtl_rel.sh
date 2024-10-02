python3.8 train_rel_ori.py --learner multi_task

# 2023-05-20 17:56:31,672 - ContinualLearningLog - INFO - Running order 1
# 23-05-20 18:18:37,287 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.1092, accuracy = 0.9153       
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:35<00:00,  7.59it/s]  
# 2023-05-20 18:21:23,244 - ContinualLearningLog - INFO - Saved the model with name Baseline-order5-id0-2023-05-20_17-56-41.448430.pt
# 2023-05-20 18:21:23,244 - ContinualLearningLog - INFO - ----------Testing starts here----------       
# 100%|███████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.84it/s]  
# 2023-05-20 18:24:00,232 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.8590               
# 2023-05-20 18:24:00,533 - ContinualLearningLog - INFO - Running order 2
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:29<00:00,  7.62it/s]  
# 2023-05-20 18:48:42,534 - ContinualLearningLog - INFO - Saved the model with name Baseline-order5-id1-2023-05-20_18-24-06.310453.pt
# 2023-05-20 18:48:42,535 - ContinualLearningLog - INFO - ----------Testing starts here----------       
# 100%|███████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.84it/s]  
# 2023-05-20 18:51:19,513 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.8474               
# 2023-05-20 18:51:19,798 - ContinualLearningLog - INFO - Running order 3
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:39<00:00,  7.57it/s]  
# 2023-05-20 19:16:12,554 - ContinualLearningLog - INFO - Saved the model with name Baseline-order5-id2-2023-05-20_18-51-26.469956.pt
# 2023-05-20 19:16:12,554 - ContinualLearningLog - INFO - ----------Testing starts here----------       
# 100%|███████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.82it/s]  
# 2023-05-20 19:18:49,677 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.8593               
# 2023-05-20 19:18:49,959 - ContinualLearningLog - INFO - Running order 4
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:30<00:00,  7.62it/s]  
# 2023-05-20 19:43:33,385 - ContinualLearningLog - INFO - Saved the model with name Baseline-order5-id3-2023-05-20_19-18-57.122864.pt
# 2023-05-20 19:43:33,385 - ContinualLearningLog - INFO - ----------Testing starts here----------       
# 100%|███████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.82it/s]  
# 2023-05-20 19:46:10,527 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.8506               
# 2023-05-20 19:46:10,813 - ContinualLearningLog - INFO - Running order 5
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:27<00:00,  7.63it/s]
# 2023-05-20 20:10:49,824 - ContinualLearningLog - INFO - Saved the model with name Baseline-order5-id4-2023-05-20_19-46-16.515198.pt
# 2023-05-20 20:10:49,824 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 100%|███████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.82it/s]
# 2023-05-20 20:13:26,942 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.8351
# 2023-05-20 20:13:27,205 - ContinualLearningLog - INFO - Average accuracy across runs: 0.8502857142857143
# 2023-05-20 20:13:27,205 - ContinualLearningLog - INFO - [TIME] End Run at 2023-05-20T20:13:27 within 2.282906242940161 hours


# python3.8 train_rel_ori.py --model roberta --learner multi_task

# 2023-05-27 07:11:59,509 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.8312               
# 2023-05-27 07:11:59,771 - ContinualLearningLog - INFO - Running order 4
# Shuffle Index:  [7, 8, 3, 0, 2, 9, 1, 4, 5, 6]
# Train Dataset Length: [1126, 3353, 558, 2259, 6212, 565, 1707, 565, 12875, 15580]
# 2023-05-27 07:37:37,215 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.1295, accuracy = 0.8942
# 2023-05-27 07:41:43,797 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.8429               
# 2023-05-27 07:41:44,070 - ContinualLearningLog - INFO - Running order 5 
# Shuffle Index:  [9, 6, 7, 3, 0, 2, 4, 8, 5, 1]
# Train Dataset Length: [558, 1707, 15580, 1126, 3353, 565, 12875, 2259, 6212, 565]
# 2023-05-27 08:02:32,062 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.1368, accuracy = 0.9025
# 2023-05-27 08:07:09,857 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.1353, accuracy = 0.8975
# 2023-05-27 08:08:06,488 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.1167, accuracy = 0.9192
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [26:13<00:00,  7.12it/s]
# 2023-05-27 08:08:07,829 - ContinualLearningLog - INFO - Saved the model with name Baseline-order5-id4-2023-05-27_07-41-52.748428.pt
# 2023-05-27 08:08:07,829 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [03:10<00:00, 14.67it/s]
# 2023-05-27 08:11:18,756 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.7990
# 2023-05-27 08:11:18,994 - ContinualLearningLog - INFO - Average accuracy across runs: 0.7156428571428572
# 2023-05-27 08:11:18,995 - ContinualLearningLog - INFO - [TIME] End Run at 2023-05-27T08:11:18 within 2.4627672443787256 hours



# Order 1
# Shuffle Index:  [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
# Train Dataset Length: [565, 1707, 2259, 12875, 6212, 558, 15580, 565, 1126, 3353]
# Order 2
# Shuffle Index: [3, 5, 2, 4, 1, 8, 7, 0, 6, 9]
# Train Dataset Length: [6212, 558, 2259, 565, 1126, 12875, 565, 3353, 15580, 1707]
# Order 3
# Shuffle Index: [7, 5, 0, 2, 4, 9, 1, 6, 3, 8]
# Train Dataset Length: [2259, 3353, 1126, 565, 558, 12875, 6212, 565, 1707, 15580]
# Order 4 
# Shuffle Index: [7, 8, 3, 0, 2, 9, 1, 4, 5, 6]
# Train Dataset Length: [1126, 3353, 558, 2259, 6212, 565, 1707, 565, 12875, 15580]
# Order 5 
# Shuffle Index: [9, 6, 7, 3, 0, 2, 4, 8, 5, 1]
# Train Dataset Length: [558, 1707, 15580, 1126, 3353, 565, 12875, 2259, 6212, 565]


# python3.8 train_rel_ori.py --model roberta --learner multi_task --lr 4e-4


#  54%|████████████████████████████████▏                           | 5999/11200 [14:16<12:49,  6.76it/s]
#  2023-05-27 12:55:22,320 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.4149, accuracy = 0.1742
#  71%|██████████████████████████████████████████▊                 | 7999/11200 [19:01<06:47,  7.86it/s]
#  2023-05-27 13:00:07,080 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.4140, accuracy = 0.1707 
# 2023-05-27 13:04:43,086 - Baseline-Log - INFO - Epoch 1 metrics: Loss = 0.4144, accuracy = 0.1628 
# RankingLabel:  [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# PRED:  [2, 9, 6]
# ANS:  [0, 0, 0]
# 2023-05-27 13:07:36,573 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [03:11<00:00, 14.62it/s]2023-05-27 13:10:48,143 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.9996
# 2023-05-27 13:10:48,406 - ContinualLearningLog - INFO - Accuracy across runs = [0.8476190476190476, 0.5301190476190476, 1.0, 1.0, 0.9996428571428572]
# 2023-05-27 13:10:48,407 - ContinualLearningLog - INFO - Average accuracy across runs: 0.8754761904761905
# 2023-05-27 13:10:48,407 - ContinualLearningLog - INFO - [TIME] End Run at 2023-05-27T13:10:48 within 2.5015899260838825 hours

# ~~~~~ After Bug Fixes~~~~~~
# 100%|███████████████████████████████████████████████████████████████████████| 11200/11200 [25:45<00:00,  7.24it/s]
# 2023-09-06 21:09:36,405 - ContinualLearningLog - INFO - Saved the model with name Baseline-order5-id4-2023-09-06_20-43-46.542806.pt
# 2023-09-06 21:09:36,410 - ContinualLearningLog - INFO - ----------Testing starts here----------                    
# 36%|██████████████████████████▍                                               | 998/2800 [01:07<02:01, 14.81it/s]
# 2023-09-06 21:10:43,607 - Baseline-Log - INFO - Epoch 1 metrics: Loss = nan, accuracy = 0.9995
#  71%|████████████████████████████████████████████████████                     | 1998/2800 [02:14<00:53, 15.05it/s]
# python3.8 train_rel_ori.py --model roberta --learner multi_task --lr 4e-4
# 2023-09-06 21:12:44,496 - Baseline-Log - INFO - Overall test metrics: Accuracy = 0.9997
# 2023-09-06 21:12:44,786 - ContinualLearningLog - INFO - Accuracy across runs = [0.8471428571428572, 0.5300892857142857, 1.0, 1.0, 0.9997321428571428]
# 2023-09-06 21:12:44,786 - ContinualLearningLog - INFO - Average accuracy across runs: 0.8753928571428571
# 2023-09-06 21:12:44,786 - ContinualLearningLog - INFO - [TIME] End Run at 2023-09-06T21:12:44 within 2.4219988587167527 hours