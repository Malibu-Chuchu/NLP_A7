# A7: Training Distillation vs LoRA
#### Submitted by: Patsachon Pattakulpong st124952 
- if you want to see whole file you can access Google Drive link: https://drive.google.com/drive/folders/1wCidZurRrMqwNbA_ES9c7_ArVG0-louI?usp=share_link
### Task 1: Hate Speech Dataset that i use: https://huggingface.co/datasets/SetFit/hate_speech18
### Task 2: Train the student model 
- 1. Using the odd layers {1, 3, 5, 7, 9, 11} from the 12-layer teacher to the 6-layer student.
  2. Train the student model using the even layers {2, 4, 6, 8, 10, 12} from the 12-layer teacher to the 6-layer student.
     Here's the code:
     <img width="1101" alt="Screenshot 2568-03-23 at 4 07 30 PM" src="https://github.com/user-attachments/assets/b7f7e38a-0b34-43ca-84eb-135224a0a5ee" />
  3. To implement LoRA for a 12-layer student model, start by loading a pre-trained BERT model using the AutoModelForSequenceClassification class. Configure the LoRA parameters with a LoraConfig object, ensuring you specify the appropriate task type as SEQ_CLS for sequence classification rather than CAUSAL_LM. Set key hyperparameters including rank (r=8), scaling factor (lora_alpha=16), and dropout rate (lora_dropout=0.1), while explicitly targeting the attention matrices ("query", "key", "value") for efficient fine-tuning. Apply this configuration using get_peft_model() and check trainable parameters with print_trainable_parameters(). Set up an AdamW optimizer with a small learning rate (2e-5) and weight decay (0.01), paired with a linear learning rate scheduler that includes a 10% warmup period. When implementing the training loop, move the model and batches to the appropriate device, calculate loss through forward passes, and update parameters with backward passes while tracking progress. This approach significantly reduces memory requirements compared to full fine-tuning while providing targeted adaptation of the model to your specific task.
  
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

#### Web Application Discussion

The implementation of the Odd Layer distillation model in our sentiment analysis web app revealed an interesting challenge in hate speech detection. Despite achieving the highest test performance (0.9020) compared to the Even Layer (0.8980) and LoRA models (0.8520), it incorrectly classifies clearly hateful content like "I hate you" as noHate.

This misclassification highlights a critical gap between test set performance metrics and real-world application. Several factors may explain this phenomenon:

1. The training data likely contained an imbalance or bias in how hate speech was represented. The model might have learned to associate hate speech with more complex or specific patterns rather than direct expressions like "I hate you."

2. Context sensitivity plays a crucial role. The Odd Layer model, while preserving important knowledge from specific layers, may have lost contextual understanding that would help it correctly classify straightforward expressions of hate.

3. The distillation process itself might have prioritized certain features over others, inadvertently diminishing the importance of common negative expressions in favor of more subtle patterns that performed well on the specific test set.

### Appendix
![output_odd](https://github.com/user-attachments/assets/de39c881-9fd1-417a-aa50-1eaddfe6eda9)
![output_lora](https://github.com/user-attachments/assets/685de207-136f-4bb8-99c3-35be8a46b1f1)
![output_even](https://github.com/user-attachments/assets/f723d245-e0d5-4db9-ab2b-a2ecbbbc3096)
![output_compare](https://github.com/user-attachments/assets/ea16f868-4076-47ea-90e4-95a7bab198ea)

