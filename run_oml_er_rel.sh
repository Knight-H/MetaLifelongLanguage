
# python3.8 train_rel_ori.py --updates 5 --write_prob 0.01 --model roberta --learner oml

# python3.8 train_rel_ori.py --updates 5 --inner_lr 0.01 --write_prob 0.01 --model roberta --learner oml
# 2023-05-16 11:18:08,456 - ContinualLearningLog - INFO - Running order 1
# 2023-05-16 11:18:21,419 - ContinualLearningLog - INFO - Using OML as learner
# 2023-05-16 11:18:21,419 - ContinualLearningLog - INFO - Generating continual learning data
# 2023-05-16 11:18:22,134 - ContinualLearningLog - INFO - Finished generating continual learning data
# 2023-05-16 11:18:22,135 - ContinualLearningLog - INFO - ----------Training starts here----------
# 2023-05-16 11:18:22,135 - OML-Log - INFO - Replay frequency: 67
# 2023-05-16 11:18:22,135 - OML-Log - INFO - Replay steps: 4
# 100%|██████████████████████████████████████████████████████████▉| 11199/11200 [10:40<00:00, 12.96it/s]2023-05-16 11:29:02,658 - OML-Log - INFO - Terminating training as all the data is seen
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [10:40<00:00, 17.49it/s]2023-05-16 11:29:03,563 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id0-2023-05-16_11-18-22.135182.pt
# 2023-05-16 11:29:03,563 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-16 11:29:03,744 - OML-Log - INFO - Support set metrics: Loss = 0.4627, accuracy = 0.5333
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.88it/s]2023-05-16 11:31:40,306 - OML-Log - INFO - Test metrics: Loss = 0.3084, accuracy = 0.2735
# 2023-05-16 11:31:40,306 - OML-Log - INFO - Overall test metrics: Accuracy = 0.2735
# 2023-05-16 11:31:40,577 - ContinualLearningLog - INFO - Running order 2
# 2023-05-16 11:31:49,104 - ContinualLearningLog - INFO - ----------Training starts here----------
# 2023-05-16 11:31:49,105 - OML-Log - INFO - Replay frequency: 67
# 2023-05-16 11:31:49,105 - OML-Log - INFO - Replay steps: 4
# 100%|██████████████████████████████████████████████████████████▉| 11199/11200 [11:42<00:00, 12.99it/s]2023-05-16 11:43:31,603 - OML-Log - INFO - Terminating training as all the data is seen
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [11:42<00:00, 15.94it/s]
# 2023-05-16 11:43:32,573 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id1-2023-05-16_11-31-49.105009.pt
# 2023-05-16 11:43:32,574 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-16 11:43:32,742 - OML-Log - INFO - Support set metrics: Loss = 0.4895, accuracy = 0.2000
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.88it/s]
# 2023-05-16 11:46:09,323 - OML-Log - INFO - Test metrics: Loss = 0.3085, accuracy = 0.1654             
# 2023-05-16 11:46:09,323 - OML-Log - INFO - Overall test metrics: Accuracy = 0.1654


# python3.8 train_rel_ori.py --updates 5 --write_prob 0.01 --model roberta --learner oml
# 2023-05-16 11:47:09,971 - ContinualLearningLog - INFO - Running order 1
# 2023-05-16 11:47:22,627 - ContinualLearningLog - INFO - ----------Training starts here----------
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [10:27<00:00, 17.84it/s]
# 2023-05-16 11:57:51,965 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-16 11:57:52,158 - OML-Log - INFO - Support set metrics: Loss = 0.6979, accuracy = 0.5333
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.86it/s]
# 2023-05-16 12:00:28,968 - OML-Log - INFO - Overall test metrics: Accuracy = 0.2740
# 2023-05-16 12:00:29,235 - ContinualLearningLog - INFO - Running order 2
# 2023-05-16 12:00:37,828 - ContinualLearningLog - INFO - ----------Training starts here----------
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [10:28<00:00, 17.81it/s]
# 2023-05-16 12:11:08,454 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-16 12:11:08,627 - OML-Log - INFO - Support set metrics: Loss = 0.4976, accuracy = 0.4000
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [02:35<00:00, 17.95it/s]
# 2023-05-16 12:13:44,602 - OML-Log - INFO - Test metrics: Loss = 0.3047, accuracy = 0.1471
# 2023-05-16 12:13:44,603 - OML-Log - INFO - Overall test metrics: Accuracy = 0.1471
# 2023-05-16 12:13:44,883 - ContinualLearningLog - INFO - Running order 3
# 2023-05-16 12:13:53,541 - ContinualLearningLog - INFO - ----------Training starts here----------
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [10:30<00:00, 17.77it/s]
# 2023-05-16 12:24:25,800 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-16 12:24:26,019 - OML-Log - INFO - Support set metrics: Loss = 0.4219, accuracy = 0.4667
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.93it/s]
# 2023-05-16 12:27:02,212 - OML-Log - INFO - Test metrics: Loss = 0.3047, accuracy = 0.2506
# 2023-05-16 12:27:02,212 - OML-Log - INFO - Overall test metrics: Accuracy = 0.2506
# 2023-05-16 12:27:02,489 - ContinualLearningLog - INFO - Running order 4
# 2023-05-16 12:27:10,964 - ContinualLearningLog - INFO - ----------Training starts here----------
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [10:13<00:00, 18.26it/s]
# 2023-05-16 12:37:29,720 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id3-2023-05-16_12-27-10.964671.pt
# 2023-05-16 12:37:29,720 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-16 12:37:29,924 - OML-Log - INFO - Support set metrics: Loss = 0.4856, accuracy = 0.2667
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.89it/s]
# 2023-05-16 12:40:06,439 - OML-Log - INFO - Test metrics: Loss = 0.3047, accuracy = 0.1192
# 2023-05-16 12:40:06,439 - OML-Log - INFO - Overall test metrics: Accuracy = 0.1192
# 2023-05-16 12:40:06,719 - ContinualLearningLog - INFO - Running order 5
# 2023-05-16 12:40:15,359 - ContinualLearningLog - INFO - ----------Training starts here----------
# 100%|███████████████████████████████████████████████████████████| 11200/11200 [10:36<00:00, 17.58it/s]
# 2023-05-16 12:50:57,458 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id4-2023-05-16_12-40-15.359877.pt
# 2023-05-16 12:50:57,458 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-16 12:50:57,676 - OML-Log - INFO - Support set metrics: Loss = 0.4162, accuracy = 0.3333
# 100%|█████████████████████████████████████████████████████████████| 2800/2800 [02:36<00:00, 17.90it/s]
# 2023-05-16 12:53:34,118 - OML-Log - INFO - Test metrics: Loss = 0.3047, accuracy = 0.2445
# 2023-05-16 12:53:34,119 - OML-Log - INFO - Overall test metrics: Accuracy = 0.2445
# 2023-05-16 12:53:34,399 - ContinualLearningLog - INFO - Average accuracy across runs: 0.20709523809523808
# 2023-05-16 12:53:34,399 - ContinualLearningLog - INFO - [TIME] End Run at 2023-05-16T12:53:34 within 1.1075384220149782 hours


python3.8 train_rel_ori.py --updates 5 --learner oml

# 2023-05-18 17:39:54,142 - ContinualLearningLog - INFO - Running order 1
# 100%|███████████████████████████████████████████████████████████████| 11200/11200 [10:37<00:00, 17.57it/s]
# 2023-05-18 17:50:47,457 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id0-2023-05-18_17-40-04.345038.pt
# 2023-05-18 17:50:47,457 - ContinualLearningLog - INFO - ----------Testing starts here----------        
# 2023-05-18 17:50:47,650 - OML-Log - INFO - Support set metrics: Loss = 0.1477, accuracy = 0.9333       
# 100%|█████████████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.80it/s]
# 2023-05-18 17:53:24,993 - OML-Log - INFO - Test metrics: Loss = 0.1728, accuracy = 0.7158              
# 2023-05-18 17:53:24,994 - OML-Log - INFO - Overall test metrics: Accuracy = 0.7158                     
# 2023-05-18 17:53:25,281 - ContinualLearningLog - INFO - Running order 2
# 100%|███████████████████████████████████████████████████████████████| 11200/11200 [10:45<00:00, 17.34it/s]
# 2023-05-18 18:04:26,195 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id1-2023-05-18_17-53-31.281001.pt
# 2023-05-18 18:04:26,195 - ContinualLearningLog - INFO - ----------Testing starts here----------        
# 2023-05-18 18:04:26,412 - OML-Log - INFO - Support set metrics: Loss = 0.1512, accuracy = 0.9333       
# 100%|█████████████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.77it/s]
# 2023-05-18 18:07:03,943 - OML-Log - INFO - Test metrics: Loss = 0.1693, accuracy = 0.6740              
# 2023-05-18 18:07:03,943 - OML-Log - INFO - Overall test metrics: Accuracy = 0.6740                     
# 2023-05-18 18:07:04,232 - ContinualLearningLog - INFO - Running order 3
# 100%|███████████████████████████████████████████████████████████████| 11200/11200 [10:39<00:00, 17.52it/s]
# 2023-05-18 18:17:56,981 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id2-2023-05-18_18-07-10.266573.pt
# 2023-05-18 18:17:56,982 - ContinualLearningLog - INFO - ----------Testing starts here----------        
# 2023-05-18 18:17:57,201 - OML-Log - INFO - Support set metrics: Loss = 0.3180, accuracy = 1.0000       
# 100%|█████████████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.77it/s]
# 2023-05-18 18:20:34,744 - OML-Log - INFO - Test metrics: Loss = 0.2366, accuracy = 0.6857              
# 2023-05-18 18:20:34,745 - OML-Log - INFO - Overall test metrics: Accuracy = 0.6857                     
# 2023-05-18 18:20:35,035 - ContinualLearningLog - INFO - Running order 4 
# 100%|███████████████████████████████████████████████████████████████| 11200/11200 [10:42<00:00, 17.42it/s]
# 2023-05-18 18:31:33,192 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id3-2023-05-18_18-20-41.494118.pt
# 2023-05-18 18:31:33,193 - ContinualLearningLog - INFO - ----------Testing starts here----------        
# 2023-05-18 18:31:33,437 - OML-Log - INFO - Support set metrics: Loss = 0.2407, accuracy = 0.7333       
# 100%|█████████████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.77it/s]
# 2023-05-18 18:34:11,047 - OML-Log - INFO - Test metrics: Loss = 0.1860, accuracy = 0.7089              
# 2023-05-18 18:34:11,048 - OML-Log - INFO - Overall test metrics: Accuracy = 0.7089                     
# 2023-05-18 18:34:11,340 - ContinualLearningLog - INFO - Running order 5
# 2023-05-18 18:34:16,353 - ContinualLearningLog - INFO - Generating continual learning data
# Train Dataset Length: [565, 558, 565, 2259, 15580, 12875, 1126, 6212, 1707, 3353]
# 2023-05-18 18:34:17,068 - ContinualLearningLog - INFO - Finished generating continual learning data
# 2023-05-18 18:34:17,069 - ContinualLearningLog - INFO - ----------Training starts here----------
# 2023-05-18 18:34:17,069 - OML-Log - INFO - Replay frequency: 67
# 2023-05-18 18:34:17,069 - OML-Log - INFO - Replay steps: 4
# 100%|███████████████████████████████████████████████████████████████| 11200/11200 [10:40<00:00, 17.49it/s]
# 2023-05-18 18:45:06,069 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id4-2023-05-18_18-34-17.069257.pt
# 2023-05-18 18:45:06,069 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-05-18 18:45:06,308 - OML-Log - INFO - Support set metrics: Loss = 0.1862, accuracy = 0.9333
# 100%|█████████████████████████████████████████████████████████████████| 2800/2800 [02:37<00:00, 17.76it/s]
# 2023-05-18 18:47:43,931 - OML-Log - INFO - Test metrics: Loss = 0.2039, accuracy = 0.6481
# 2023-05-18 18:47:43,931 - OML-Log - INFO - Overall test metrics: Accuracy = 0.6481
# 2023-05-18 18:47:44,228 - ContinualLearningLog - INFO - Average accuracy across runs: 0.6865238095238094
# 2023-05-18 18:47:44,228 - ContinualLearningLog - INFO - [TIME] End Run at 2023-05-18T18:47:44 within 1.1313476826084985 hours


# Roberta with version 2.5?  Copy Inner LR from MelSTA since it should be similar... Or just train with the MeLSTA code. Same thing!
python3.8 train_rel_ori.py --updates 5 --inner_lr 0.005 --learner oml --model roberta --write_prob 0.01 


# 2023-09-14 13:12:41,356 - ContinualLearningLog - INFO - Shuffle Index: [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
# 2023-09-14 13:32:31,560 - ContinualLearningLog - INFO - Shuffle Index: [8, 0, 2, 5, 3, 1, 6, 9, 4, 7]
# 2023-09-14 13:52:14,838 - ContinualLearningLog - INFO - Shuffle Index: [3, 6, 9, 1, 0, 5, 8, 4, 7, 2]
# 2023-09-14 14:12:01,328 - ContinualLearningLog - INFO - Shuffle Index: [9, 0, 5, 2, 7, 8, 4, 6, 3, 1]
# 2023-09-14 14:31:35,551 - ContinualLearningLog - INFO - Shuffle Index: [0, 8, 1, 9, 6, 5, 4, 3, 2, 7]


# 2023-09-14 14:12:01,328 - ContinualLearningLog - INFO - Shuffle Index: [9, 0, 5, 2, 7, 8, 4, 6, 3, 1]     
# 2023-09-14 14:12:01,329 - ContinualLearningLog - INFO - Train Dataset Length: [12875, 1707, 1126, 565, 3353, 2259, 6212, 558, 15580, 565]
# 2023-09-14 14:12:01,329 - ContinualLearningLog - INFO - Finished generating continual learning data       
# 2023-09-14 14:12:01,329 - ContinualLearningLog - INFO - ----------Training starts here----------          
# 2023-09-14 14:12:01,330 - OML-Log - INFO - Replay frequency: 67                                           
# 2023-09-14 14:12:01,330 - OML-Log - INFO - Replay steps: 4                                                
# 100%|███████████████████████████████████████████████████████████████████████████████████| 11200/11200 [15:16<00:00,  8.28it/s]2023-09-14 14:27:17,418 - OML-Log - INFO - Terminating training as all the data is seen        
# 100%|███████████████████████████████████████████████████████████████████████████████████| 11200/11200 [15:16<00:00, 12.23it/s] 
# 2023-09-14 14:27:18,665 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id3-2023-09-14_14-12-01.330229.pt  
# 2023-09-14 14:27:18,665 - ContinualLearningLog - INFO - ----------Testing starts here----------           
# 2023-09-14 14:27:19,031 - OML-Log - INFO - Support set metrics: Loss = 0.3720, accuracy = 0.9000          
# 100%|█████████████████████████████████████████████████████████████████████████████████████| 2800/2800 [04:00<00:00, 11.66it/s] 
# 2023-09-14 14:31:19,098 - OML-Log - INFO - Test metrics: Loss = 0.2156, accuracy = 0.6786                 
# 2023-09-14 14:31:19,098 - OML-Log - INFO - Overall test metrics: Accuracy = 0.6786                        
# 2023-09-14 14:31:19,468 - ContinualLearningLog - INFO - Running order 5
# 2023-09-14 14:31:34,369 - ContinualLearningLog - INFO - Generating continual learning data                
# 2023-09-14 14:31:35,551 - ContinualLearningLog - INFO - Shuffle Index: [0, 8, 1, 9, 6, 5, 4, 3, 2, 7] 
# 2023-09-14 14:31:35,552 - ContinualLearningLog - INFO - ----------Training starts here----------
# 2023-09-14 14:31:35,552 - OML-Log - INFO - Replay frequency: 67
# 2023-09-14 14:31:35,552 - OML-Log - INFO - Replay steps: 4
# 100%|███████████████████████████████████████████████████████████████████████████████████| 11200/11200 [15:37<00:00,  8.29it/s]2023-09-14 14:47:12,830 - OML-Log - INFO - Terminating training as all the data is seen
# 100%|███████████████████████████████████████████████████████████████████████████████████| 11200/11200 [15:37<00:00, 11.95it/s]
# 2023-09-14 14:47:13,857 - ContinualLearningLog - INFO - Saved the model with name OML-order5-id4-2023-09-14_14-31-35.552733.pt
# 2023-09-14 14:47:13,858 - ContinualLearningLog - INFO - ----------Testing starts here----------
# 2023-09-14 14:47:14,166 - OML-Log - INFO - Support set metrics: Loss = 0.3933, accuracy = 0.8500
# 100%|█████████████████████████████████████████████████████████████████████████████████████| 2800/2800 [03:58<00:00, 11.73it/s]
# 2023-09-14 14:51:12,778 - OML-Log - INFO - Test metrics: Loss = 0.2788, accuracy = 0.6598
# 2023-09-14 14:51:12,779 - OML-Log - INFO - Overall test metrics: Accuracy = 0.6598
# 2023-09-14 14:51:13,165 - ContinualLearningLog - INFO - Accuracy across runs = [0.5772321428571429, 0.7144642857142857, 0.6467857142857143, 0.6785714285714286, 0.6598214285714286]
# 2023-09-14 14:51:13,166 - ContinualLearningLog - INFO - Average accuracy across runs: 0.655375
# 2023-09-14 14:51:13,166 - ContinualLearningLog - INFO - [TIME] End Run at 2023-09-14T14:51:13 within 1.6494792284568152 hours


# WRONG SHUFFLE INDEX! RUN AGAIN!
# Try make random shuffle generator
# 2023-09-14 16:46:56,887 - ContinualLearningLog - INFO - Shuffle Index: [7, 3, 2, 8, 5, 6, 9, 4, 0, 1]
# 2023-09-14 17:06:38,618 - ContinualLearningLog - INFO - Shuffle Index: [3, 5, 2, 4, 1, 8, 7, 0, 6, 9]
# 2023-09-14 17:26:31,413 - ContinualLearningLog - INFO - Shuffle Index: [7, 5, 0, 2, 4, 9, 1, 6, 3, 8]
# 2023-09-14 17:46:23,256 - ContinualLearningLog - INFO - Shuffle Index: [7, 8, 3, 0, 2, 9, 1, 4, 5, 6]
# 2023-09-14 18:06:13,105 - ContinualLearningLog - INFO - Shuffle Index: [9, 6, 7, 3, 0, 2, 4, 8, 5, 1]