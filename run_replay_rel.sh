python3.8 train_rel_ori.py --learner replay

# 2023-05-21 00:11:41,304 - ContinualLearningLog - INFO - Running order 1
# 2023-05-21 00:33:06,043 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0863, accuracy = 0.8928     
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:55<00:00,  7.49it/s]
# 2023-05-21 00:36:52,882 - ContinualLearningLog - INFO - Saved the model with name Replay-order5-id0-2023-05-21_00-11-51.907188.pt
# 2023-05-21 00:36:52,882 - ContinualLearningLog - INFO - ----------Testing starts here----------     
# 2023-05-21 00:39:28,828 - Replay-Log - INFO - Overall test metrics: Accuracy = 0.5748               
# 2023-05-21 00:39:29,151 - ContinualLearningLog - INFO - Running order 2
# 2023-05-21 01:00:45,149 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0876, accuracy = 0.8863     
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:50<00:00,  7.51it/s]
# 2023-05-21 01:04:30,263 - ContinualLearningLog - INFO - Saved the model with name Replay-order5-id1-2023-05-21_00-39-35.431126.pt
# 2023-05-21 01:04:30,263 - ContinualLearningLog - INFO - ----------Testing starts here----------     
# 2023-05-21 01:07:06,650 - Replay-Log - INFO - Overall test metrics: Accuracy = 0.6325               
# 2023-05-21 01:07:06,941 - ContinualLearningLog - INFO - Running order 3
# 2023-05-21 01:07:12,124 - ContinualLearningLog - INFO - Generating continual learning data          Train Dataset Length: [15580, 6212, 1707, 565, 565, 1126, 558, 2259, 3353, 12875]
# 2023-05-21 01:28:21,103 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0605, accuracy = 0.9167     
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [24:53<00:00,  7.50it/s]
# 2023-05-21 01:32:11,057 - ContinualLearningLog - INFO - Saved the model with name Replay-order5-id2-2023-05-21_01-07-12.853198.pt
# 2023-05-21 01:32:11,063 - ContinualLearningLog - INFO - ----------Testing starts here----------     
# 2023-05-21 01:34:47,134 - Replay-Log - INFO - Overall test metrics: Accuracy = 0.5889               
# 2023-05-21 01:34:47,419 - ContinualLearningLog - INFO - Running order 4
# 2023-05-21 01:56:30,744 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0530, accuracy = 0.9463     
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [25:19<00:00,  7.37it/s]
# 2023-05-21 02:00:17,824 - ContinualLearningLog - INFO - Saved the model with name Replay-order5-id3-2023-05-21_01-34-53.617859.pt
# 2023-05-21 02:00:17,825 - ContinualLearningLog - INFO - ----------Testing starts here----------     
# 2023-05-21 02:02:54,181 - Replay-Log - INFO - Overall test metrics: Accuracy = 0.6382               
# 2023-05-21 02:02:54,458 - ContinualLearningLog - INFO - Running order 5
# 2023-05-21 02:24:43,858 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0577, accuracy = 0.9410
# 100%|█████████████████████████████████████████████████████████| 11200/11200 [25:25<00:00,  7.34it/s]
# 2023-05-21 02:28:33,947 - ContinualLearningLog - INFO - Saved the model with name Replay-order5-id4-2023-05-21_02-03-00.550031.pt
# 2023-05-21 02:28:33,948 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-21 02:31:10,340 - Replay-Log - INFO - Overall test metrics: Accuracy = 0.6438
# 2023-05-21 02:31:10,632 - ContinualLearningLog - INFO - Average accuracy across runs: 0.6156428571428572
# 2023-05-21 02:31:10,633 - ContinualLearningLog - INFO - [TIME] End Run at 2023-05-21T02:31:10 within 2.3256059674421947 hours

# After Changing Fix Index!
python3.8 train_rel_ori.py --learner replay  --model roberta --lr 1e-5

# 2023-09-15 18:28:02,145 - ContinualLearningLog - INFO - Shuffle Index: [3, 5, 2, 4, 1, 8, 7, 0, 6, 9]                   
# 2023-09-15 18:28:02,146 - ContinualLearningLog - INFO - Train Dataset Length: [6212, 558, 2259, 565, 1126, 12875, 565, 3353, 15580, 1707]
# 2023-09-15 18:28:02,146 - ContinualLearningLog - INFO - Finished generating continual learning data                     
# 2023-09-15 18:28:02,146 - ContinualLearningLog - INFO - ----------Training starts here----------                         
# 18%|███████████▍                                                    | 1998/11200 [02:30<13:03, 11.74it/s]
# 2023-09-15 18:30:32,871 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.2639, accuracy = 0.8888                                        
# 36%|██████████████████████▊                                         | 3999/11200 [06:17<14:44,  8.15it/s]
# 2023-09-15 18:34:20,448 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.1298, accuracy = 0.9085                                        
# 54%|██████████████████████████████████▎                             | 5999/11200 [10:38<11:43,  7.39it/s]
# 2023-09-15 18:38:41,190 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.1035, accuracy = 0.9293                                        
# 71%|█████████████████████████████████████████████▋                  | 7999/11200 [16:27<10:18,  5.18it/s]
# 2023-09-15 18:44:29,918 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.1055, accuracy = 0.8650                                        
# 89%|█████████████████████████████████████████████████████████▏      | 9999/11200 [23:03<03:59,  5.02it/s]
# 2023-09-15 18:51:06,086 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0909, accuracy = 0.8671                                       
# 100%|███████████████████████████████████████████████████████████████| 11200/11200 [27:03<00:00,  6.90it/s]              
# 2023-09-15 18:55:06,102 - ContinualLearningLog - INFO - Saved the model with name Replay-order5-id1-2023-09-15_18-28-02.146308.pt
# 2023-09-15 18:55:06,102 - ContinualLearningLog - INFO - ----------Testing starts here----------                         
# 2023-09-15 18:58:20,837 - Replay-Log - INFO - Overall test metrics: Accuracy = 0.6035
# 2023-09-15 19:58:56,033 - ContinualLearningLog - INFO - Finished generating continual learning data
# 2023-09-15 19:58:56,033 - ContinualLearningLog - INFO - ----------Training starts here----------
#  18%|███████████▍                                                    | 1999/11200 [03:14<15:55,  9.63it/s]2023-09-15 20:02:10,952 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.2871, accuracy = 0.8049
#  36%|██████████████████████▊                                         | 3999/11200 [06:46<12:45,  9.40it/s]2023-09-15 20:05:42,829 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.1995, accuracy = 0.8681
#  54%|██████████████████████████████████▎                             | 5999/11200 [10:49<13:57,  6.21it/s]2023-09-15 20:09:46,238 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.1189, accuracy = 0.9167
#  71%|█████████████████████████████████████████████▋                  | 7999/11200 [16:13<08:21,  6.39it/s]2023-09-15 20:15:10,035 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0803, accuracy = 0.9160
#  89%|█████████████████████████████████████████████████████████▏      | 9999/11200 [21:59<03:52,  5.16it/s]2023-09-15 20:20:55,820 - Replay-Log - INFO - Epoch 1 metrics: Loss = 0.0552, accuracy = 0.9463
# 100%|███████████████████████████████████████████████████████████████| 11200/11200 [25:59<00:00,  7.18it/s]
# 2023-09-15 20:24:56,163 - ContinualLearningLog - INFO - Saved the model with name Replay-order5-id4-2023-09-15_19-58-56.033321.pt
# 2023-09-15 20:24:56,163 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-09-15 20:28:08,302 - Replay-Log - INFO - Overall test metrics: Accuracy = 0.6696
# 2023-09-15 20:28:08,575 - ContinualLearningLog - INFO - Accuracy across runs = [0.5946428571428571, 0.6034821428571429,
# 0.5496428571428571, 0.5201785714285714, 0.6696428571428571]
# 2023-09-15 20:28:08,575 - ContinualLearningLog - INFO - Average accuracy across runs: 0.5875178571428571
# 2023-09-15 20:28:08,575 - ContinualLearningLog - INFO - [TIME] End Run at 2023-09-15T20:28:08 within 2.5583413567807938
# hours