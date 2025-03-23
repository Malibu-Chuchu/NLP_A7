# A7: Training Distillation vs LoRA
#### Submitted by: Patsachon Pattakulpong st124952 
### Task 1: Hate Speech Dataset that i use: https://huggingface.co/datasets/SetFit/hate_speech18
### Task 2: Train the student model 
- 1. Using the odd layers {1, 3, 5, 7, 9, 11} from the 12-layer teacher to the 6-layer student.
  2. Train the student model using the even layers {2, 4, 6, 8, 10, 12} from the 12-layer teacher to the 6-layer student.
     Here's the code:
     <img width="1101" alt="Screenshot 2568-03-23 at 4 07 30 PM" src="https://github.com/user-attachments/assets/b7f7e38a-0b34-43ca-84eb-135224a0a5ee" />
     
### Task 3: Evaluation and Analysis

| Model Type  | Training Loss | Test Set Performance |
|-------------|--------------|----------------------|
| Odd Layer   |      0.2944       |        0.9020              |
| Even Layer  |0.2849              |  0.8980                    |
| LoRA        |  0.4116            |          0.8520            |
#### Evaluate Three Models
- #### Test Set Performance Analysis
- 1. The evaluation of our three fine-tuning approaches reveals significant performance differences. The Odd Layer distillation model achieved the highest test performance at 0.9020, marginally outperforming.
  2. Even Layer model's 0.8980 score. Both distillation approaches substantially outperformed the LoRA fine-tuning method, which scored 0.8520 on the test set.
     - This performance gap suggests that distillation-based approaches better preserved the model's knowledge and generalization capabilities during fine-tuning. The Odd Layer configuration's slight edge over the Even Layer model indicates that critical information might be disproportionately stored in odd-numbered layers of the original architecture.
  3. LoRA's lower performance, despite its parameter efficiency, demonstrates the tradeoff between computational savings and model effectiveness. The higher training loss (0.4116) for LoRA compared to distillation methods (0.2944 and 0.2849) further supports this conclusion, indicating that LoRA struggled more during the optimization process.

These results suggest that for applications prioritizing performance over parameter efficiency, distillation-based fine-tuning approaches—particularly targeting odd-numbered layers—would be the recommended strategy.
#### Disscussion
- **Challenges in Distillation vs. LoRA Fine-tuning**
- 1. Distillation fine-tuning models demonstrated superior performance over LoRA, with odd-layer configuration achieving the highest test accuracy at 0.9020. The primary challenge with distillation was effectively selecting which layers to retain while maintaining knowledge transfer integrity. LoRA implementation struggled with significantly higher training loss (0.4116) and lower test performance (0.8520), suggesting difficulties in parameter-efficient adaptation. 

- **To improve distillation models**, implementing importance-based layer selection rather than simple odd/even patterns would likely enhance performance further. For LoRA, adopting adaptive rank selection techniques would address the higher loss issues by allocating different ranks to layers based on their relative importance. Combining both approaches through targeted hybrid implementation—applying LoRA to sensitive layers and distillation to others—could potentially leverage the strengths of each method while mitigating their respective weaknesses.

### Task 4: Web Application
<img width="1440" alt="Screenshot 2568-03-23 at 5 10 12 PM" src="https://github.com/user-attachments/assets/14b93a9f-b112-490d-aefd-ed0953ce2561" />
<img width="1440" alt="Screenshot 2568-03-23 at 5 10 24 PM" src="https://github.com/user-attachments/assets/80fe071a-83ed-490e-8b2b-bce9b5b4b9ea" />
<img width="1440" alt="Screenshot 2568-03-23 at 5 10 34 PM" src="https://github.com/user-attachments/assets/037423fb-9ac0-4191-af26-6e244c995ea6" />
Link to my VDO: https://youtu.be/3aZegzbznyk
