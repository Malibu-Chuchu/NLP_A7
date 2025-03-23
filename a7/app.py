from flask import Flask, render_template, request
import torch
from torch.nn import Module
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel

app = Flask(__name__)

# Choose the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("./tokenizer")
teacher_model = AutoModelForSequenceClassification.from_pretrained("./teacher_model")

teacher_config_dict = teacher_model.config.to_dict()
teacher_config_dict['num_hidden_layers'] //= 2
student_config = BertConfig.from_dict(teacher_config_dict)


model = type(teacher_model)(student_config)

# Distillation
def distill_bert_weights(
    teacher: Module,
    student: Module,
    layer: str = 'odd'
) -> None:
    """
    Recursively copies the weights of the teacher to the student.
    This function is meant to be first called on a BertFor... model, but is then called on every child module recursively.
    For the encoder, only half of the teacher's layers are copied, based on the 'layer' parameter:
    - 'odd': Copies layers {1, 3, 5, 7, 9, 11} (indices 0, 2, 4, 6, 8, 10).
    - 'even': Copies layers {2, 4, 6, 8, 10, 12} (indices 1, 3, 5, 7, 9, 11).
    """
    # If the part is an entire BERT model or a BERTFor..., unpack and iterate
    if isinstance(teacher, BertModel) or type(teacher).__name__.startswith('BertFor'):
        for teacher_part, student_part in zip(teacher.children(), student.children()):
            distill_bert_weights(teacher_part, student_part, layer=layer)
    # Else if the part is an encoder, copy the specified layers
    elif isinstance(teacher, BertEncoder):
        teacher_encoding_layers = [layer for layer in next(teacher.children())]  # 12 layers
        student_encoding_layers = [layer for layer in next(student.children())]  # 6 layers
        for i in range(len(student_encoding_layers)):
            if layer == 'odd':
                teacher_index = 2 * i  # Indices: 2, 4, 6, 8, 10, 12
            elif layer == 'even':
                teacher_index = 2 * i + 1  # Indices: 1, 3, 5, 7, 9, 11
            else:
                raise ValueError("Invalid layer parameter. Must be 'odd' or 'even'.")
            student_encoding_layers[i].load_state_dict(teacher_encoding_layers[teacher_index].state_dict())
    # Else the part is a head or something else, copy the state_dict
    else:
        student.load_state_dict(teacher.state_dict())
    return student

odd_model = distill_bert_weights(teacher_model, student=model, layer='odd')
model = odd_model

model.load_state_dict(torch.load('./model/odd_model.pt', map_location=device))

model.eval()
model.to(device)

label_mapping = {0: 'noHate', 1: 'hate', 2: 'idk/skip', 3: 'relation'}

@app.route("/", methods=["GET", "POST"])
def index():
    response = None
    prompt = ""

    if request.method == "POST":
        prompt = request.form.get("prompt", "")
        if prompt:
            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt")
            # Move to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            
            # Pick the highest logit
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            response = label_mapping[predicted_class_id]
            print("Predicted class ID:", predicted_class_id)

    return render_template("index.html", prompt=prompt, response=response)

if __name__ == "__main__":
    app.run(debug=True)
