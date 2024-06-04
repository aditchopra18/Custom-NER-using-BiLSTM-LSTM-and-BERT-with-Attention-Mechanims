# Importing the training model
import Code.LSTM.Training_Model as tr_mod
import torch.utils.data
from sklearn.metrics import classification_report

# Load the test dataset
test_file = '../../Data/NCBItestset_corpus.txt'
test_lines = tr_mod.read_dataset(test_file)
test_paragraphs = tr_mod.parse_dataset(test_lines)

test_sentences = []
test_tags = []

for paragraph in test_paragraphs:
    sentences, annotations = tr_mod.parse_paragraph(paragraph)
    tagged_sentences = tr_mod.tag_annotations(sentences, annotations)
    for sentence, tags in tagged_sentences:
        test_sentences.append(sentence)
        test_tags.append(tags)

# Loading the model
model_name = 'NER_model.pth'
model = tr_mod.NERModel(len(tr_mod.word_encoder), len(tr_mod.tag_encoder.classes_)).to(tr_mod.device)
model.load_state_dict(torch.load(model_name))
model.eval()

# Prepare the test data
test_dataset = tr_mod.NERDataset(test_sentences, test_tags, tr_mod.word_encoder, tr_mod.tag_encoder, '<UNK>')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: x)

# Evaluate the model and save results
result_file = 'Test_Results.txt'
all_true_labels = []
all_pred_labels = []

with open(result_file, 'w') as file:
    with torch.no_grad():
        for batch in test_dataloader:
            sentences, tags = zip(*batch)
            sentences = torch.nn.utils.rnn.pad_sequence(sentences, batch_first=True).to(tr_mod.device)
            tags = torch.nn.utils.rnn.pad_sequence(tags, batch_first=True, padding_value=-100).to(tr_mod.device)

            outputs = model(sentences)
            outputs = outputs.view(-1, outputs.shape[-1])
            tags = tags.view(-1)

            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = tags.cpu().numpy()

            mask = true_labels != -100
            pred_labels = predictions[mask]
            true_labels = true_labels[mask]

            pred_labels_decoded = tr_mod.tag_encoder.inverse_transform(pred_labels)
            true_labels_decoded = tr_mod.tag_encoder.inverse_transform(true_labels)

            for true_label, pred_label in zip(true_labels_decoded, pred_labels_decoded):
                file.write(f'True: {true_label}, Pred: {pred_label}\n')
                all_true_labels.append(true_label)
                all_pred_labels.append(pred_label)

# Printing classification report
report = classification_report(all_true_labels, all_pred_labels)
print(report)