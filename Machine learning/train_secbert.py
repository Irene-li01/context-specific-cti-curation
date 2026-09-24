import json
import torch
from pathlib import Path
from torch import nn
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from collections import Counter

# ==========================================
# 1. Custom Dataset with Label Alignment
# ==========================================
class CTINERDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_file)
        
        # Automatically build label mappings from the dataset
        self.label_list = self._get_label_list()
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}

    def _load_data(self, data_file):
        print(f"Loading training data from {data_file}...")
        with open(data_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _get_label_list(self):
        labels = set()
        for item in self.data:
            for tag in item.get('ner_tags', []):
                labels.add(tag)
        label_list = sorted(list(labels))
        if 'O' in label_list:
            label_list.remove('O')
            label_list.insert(0, 'O')
        print(f"Discovered {len(label_list)} unique labels: {label_list}")
        return label_list

    def calculate_class_weights(self):
        """
        Dynamically calculates class weights to penalize the model for over-predicting 'O'.
        Rare entities get higher weights, 'O' gets a very low weight.
        """
        all_tags = []
        for item in self.data:
            all_tags.extend(item.get('ner_tags', []))
        
        tag_counts = Counter(all_tags)
        total_tags = sum(tag_counts.values())
        
        weights = []
        for label in self.label_list:
            count = tag_counts.get(label, 1) 
            weight = total_tags / (len(self.label_list) * count)
            if label == 'O':
                weights.append(1.0)
            else:
                weights.append(min(weight, 8.0))
                
        print(f"Calculated Class Weights: {weights}")
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item['tokens']
        tags = item['ner_tags']

        # Tokenize words. is_split_into_words=True is crucial for NER.
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        # Remove the batch dimension added by return_tensors="pt"
        item_encodings = {key: val.squeeze(0) for key, val in encoding.items()}
        
        # Align labels with tokens (subword tokenization creates extra tokens)
        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) get -100 so they are ignored in loss calculation
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First token of a given word
                label_ids.append(self.label2id[tags[word_idx]])
            else:
                # Subsequent subword tokens of the same word (can be labeled or ignored)
                label_ids.append(-100) 
            previous_word_idx = word_idx

        item_encodings['labels'] = torch.tensor(label_ids)
        return item_encodings

# ==========================================
# 2. Custom Trainer to Apply Weights
# ==========================================
class WeightedNERTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Move weights to the same device as the model (CPU/GPU)
        weights = self.class_weights.to(model.device)
        
        # Apply the weighted Cross Entropy Loss
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        
        # Only compute loss on active tokens (ignore -100)
        active_loss = labels.view(-1) != -100
        active_logits = logits.view(-1, model.config.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        
        loss = loss_fct(active_logits, active_labels)
        return (loss, outputs) if return_outputs else loss

# ==========================================
# 3. Main Training Execution Pipeline
# ==========================================
def main():
    # Model configuration
    model_name = "jackaduma/SecBERT"
    # Combined CyNER + existing data -- generated by prepare_cyner.py
    data_file = str(Path(__file__).resolve().parent / "combined_train_data.json")
    output_dir = str(Path(__file__).resolve().parent / "secbert_cti_model_final")

    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Initialize Dataset
    print("Preparing dataset and calculating weights...")
    train_dataset = CTINERDataset(data_file, tokenizer)
    class_weights = train_dataset.calculate_class_weights()

    print("Initializing model...")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(train_dataset.label_list),
        id2label=train_dataset.id2label,
        label2id=train_dataset.label2id
    )

    # Training Arguments - Optimized to prevent the "lazy model" issue
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="no",             # Change to "epoch" if you add an eval_dataset
        learning_rate=2e-5,             # Slightly lower LR for stable fine-tuning
        per_device_train_batch_size=32, # RTX 5080 16GB -- bumped up for speed
        num_train_epochs=5,             # Reduced from 15 to prevent overfitting on 4,856 samples
        weight_decay=0.01,
        save_strategy="epoch",          # Save a checkpoint every epoch
        logging_steps=10,
        push_to_hub=False,
    )

    # Data Collator for dynamic padding
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Initialize the custom weighted trainer
    trainer = WeightedNERTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,  # <--- 改成这个
    )

    print("Starting the training process...")
    trainer.train()

    print(f"Training complete. Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    print("Model successfully saved. Ready for inference!")

if __name__ == "__main__":
    main()