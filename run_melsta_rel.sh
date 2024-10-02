# OML ER in REL
# python3.8 train_rel_ori.py --updates 5 --learner oml

# replay comes from replay_steps = 5, so replay_every = 5* minibatch (4) / replay_rate (0.01) = 2000 

python3.8 train_rel_ori.py --updates 5 --inner_lr 0.005 --selective_replay --task_aware --reverse_support --write_prob 0.1 --replay_every 2000 --curriculum_replay --learner melsta --model roberta --pln 1fc


# 2023-06-16 20:19:31,400 - ContinualLearningLog - INFO - Shuffle Index: [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
# 2023-06-16 21:18:19,064 - ContinualLearningLog - INFO - Shuffle Index: [6, 5, 0, 4, 3, 7, 1, 9, 8, 2]
# 2023-06-16 22:17:23,071 - ContinualLearningLog - INFO - Shuffle Index: [4, 3, 2, 0, 1, 8, 6, 9, 7, 5]
# 2023-06-16 23:16:28,386 - ContinualLearningLog - INFO - Shuffle Index: [1, 3, 8, 5, 2, 4, 7, 0, 9, 6]
# 2023-06-17 00:14:31,729 - ContinualLearningLog - INFO - Shuffle Index: [5, 2, 3, 7, 4, 8, 0, 1, 9, 6]

# 2023-06-17 01:12:19,984 - MeLSTA-Log - INFO - Test metrics: Loss = 0.0037, accuracy = 1.0000              
# Testing on LifelongFewRelDataset
# 2023-06-17 01:12:50,158 - MeLSTA-Log - INFO - Test metrics: Loss = 0.2169, accuracy = 0.8142
# 2023-06-17 01:12:50,158 - MeLSTA-Log - INFO - [Acc] [0.9555555555555556, 0.6626356589147286, 0.711645101663586, 0.7262773722627737, 0.9788732394366197, 0.7507462686567165, 0.7674144037780402, 0.7956989247311828,
# 1.0, 0.8142493638676844]
# 2023-06-17 01:12:50,158 - MeLSTA-Log - INFO - Overall test metrics: Accuracy = 0.8163
# 2023-06-17 01:12:50,436 - ContinualLearningLog - INFO - Accuracy across runs = [0.8600202474340465, 0.8692090502246945, 0.8620806741573794, 0.8128138362377249, 0.8163095888866888]
# 2023-06-17 01:12:50,436 - ContinualLearningLog - INFO - Average accuracy across runs: 0.8440866793881069
# 2023-06-17 01:12:50,436 - ContinualLearningLog - INFO - [TIME] End Run at 2023-06-17T01:12:50 within 4.892964553104506 hours


# Fixed Index
# 2023-09-15 01:01:04,633 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.4198, accuracy = 0.8500    
# 2023-09-15 01:01:04,954 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.2570, accuracy = 1.0000    
# 2023-09-15 01:01:05,292 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.3745, accuracy = 0.9000    
# 2023-09-15 01:01:05,647 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.3022, accuracy = 0.9500    
# 2023-09-15 01:01:05,998 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.4306, accuracy = 0.8500    
# 2023-09-15 01:01:06,348 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.3976, accuracy = 1.0000    
# 2023-09-15 01:01:06,715 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.3270, accuracy = 0.9500    
# 2023-09-15 01:01:07,067 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.3299, accuracy = 0.9000    
# 2023-09-15 01:01:07,452 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.3445, accuracy = 0.8000    
# 2023-09-15 01:01:07,524 - MeLSTA-Log - INFO - Test metrics: Loss = 0.2178, accuracy = 0.7875           
# Testing on LifelongFewRelDataset 
# 2023-09-15 01:04:18,230 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1980, accuracy = 0.9000
# 2023-09-15 01:04:18,741 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1537, accuracy = 1.0000
# 2023-09-15 01:04:19,256 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1924, accuracy = 0.9000
# 2023-09-15 01:04:19,772 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1899, accuracy = 0.8500
# 2023-09-15 01:04:20,291 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1838, accuracy = 0.9500
# 2023-09-15 01:04:20,807 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1812, accuracy = 0.9500
# 2023-09-15 01:04:21,330 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.2013, accuracy = 0.9000
# 2023-09-15 01:04:21,853 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1584, accuracy = 0.9500    
# 2023-09-15 01:04:22,377 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.2283, accuracy = 0.8000
# 2023-09-15 01:04:22,464 - MeLSTA-Log - INFO - Test metrics: Loss = 0.1533, accuracy = 0.9348
# Testing on LifelongFewRelDataset
# 2023-09-15 01:04:29,005 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1117, accuracy = 0.9500    
# 2023-09-15 01:04:29,333 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1248, accuracy = 0.9000    
# 2023-09-15 01:04:29,669 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0960, accuracy = 0.9000    
# 2023-09-15 01:04:29,994 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0845, accuracy = 0.9000    
# 2023-09-15 01:04:30,315 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1353, accuracy = 0.9500    
# 2023-09-15 01:04:30,645 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1365, accuracy = 0.9000    
# 2023-09-15 01:04:30,964 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1151, accuracy = 0.9000    
# 2023-09-15 01:04:31,300 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1223, accuracy = 0.9000    
# 2023-09-15 01:04:31,626 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1791, accuracy = 0.9000    
# 2023-09-15 01:04:31,964 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1220, accuracy = 0.9000    
# 2023-09-15 01:04:32,291 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1311, accuracy = 0.9000    
# 2023-09-15 01:04:32,612 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0980, accuracy = 0.9500    
# 2023-09-15 01:04:32,935 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1113, accuracy = 0.9500    
# 2023-09-15 01:04:33,267 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1059, accuracy = 0.9500    
# 2023-09-15 01:04:33,604 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1377, accuracy = 0.9000    
# 2023-09-15 01:04:33,671 - MeLSTA-Log - INFO - Test metrics: Loss = 0.1805, accuracy = 0.8444           
# Testing on LifelongFewRelDataset     
# 2023-09-15 01:04:56,640 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0142, accuracy = 1.0000
# 2023-09-15 01:04:56,900 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0187, accuracy = 1.0000
# 2023-09-15 01:04:57,159 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0447, accuracy = 1.0000
# 2023-09-15 01:04:57,424 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1295, accuracy = 1.0000
# 2023-09-15 01:04:57,680 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0696, accuracy = 1.0000
# 2023-09-15 01:04:57,944 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0513, accuracy = 1.0000
# 2023-09-15 01:04:58,196 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0411, accuracy = 1.0000
# 2023-09-15 01:04:58,454 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0540, accuracy = 1.0000
# 2023-09-15 01:04:58,712 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.1405, accuracy = 1.0000
# 2023-09-15 01:04:58,974 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0310, accuracy = 1.0000
# 2023-09-15 01:04:59,236 - MeLSTA-Log - INFO - Support set metrics: Loss = 0.0409, accuracy = 1.0000
# 2023-09-15 01:04:59,265 - MeLSTA-Log - INFO - Test metrics: Loss = 0.0747, accuracy = 0.9237
# 2023-09-15 01:04:59,266 - MeLSTA-Log - INFO - [Acc] [1.0, 0.6982945736434109, 0.8724584103512015, 0.9416058394160584, 0.9647887323943662, 0.6358208955223881, 0.7874852420306966, 0.9348118279569892, 0.8444444444444444, 0.9236641221374046]
# 2023-09-15 01:04:59,266 - MeLSTA-Log - INFO - Overall test metrics: Accuracy = 0.8603
# 2023-09-15 01:04:59,654 - ContinualLearningLog - INFO - Accuracy across runs = [0.8544129094513966, 0.860497122883823, 0.7720312588749785, 0.7548832222915857, 0.860337408789696]
# 2023-09-15 01:04:59,654 - ContinualLearningLog - INFO - Average accuracy across runs: 0.8204323844582959
# 2023-09-15 01:04:59,658 - ContinualLearningLog - INFO - [TIME] End Run at 2023-09-15T01:04:59 within 6.058180172840754 hours