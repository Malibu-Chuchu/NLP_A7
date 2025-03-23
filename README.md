#### A7: Training Distillation vs LoRA
#### Submitted by: Patsachon Pattakulpong st124952 
### Task 1: Hate Speech Dataset that i use: https://huggingface.co/datasets/SetFit/hate_speech18
### Task 2: Train the student model 
- 1. Using the odd layers {1, 3, 5, 7, 9, 11} from the 12-layer teacher to the 6-layer student.
  2. Train the student model using the even layers {2, 4, 6, 8, 10, 12} from the 12-layer teacher to the 6-layer student.
     Here's the code:
     <img width="1101" alt="Screenshot 2568-03-23 at 4 07 30 PM" src="https://github.com/user-attachments/assets/b7f7e38a-0b34-43ca-84eb-135224a0a5ee" />
### Task 3: LoRA (Low-Rank Adaptation)
- Implement to train 12 layer student.
  
