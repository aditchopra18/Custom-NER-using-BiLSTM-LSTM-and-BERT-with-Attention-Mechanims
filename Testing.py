import Training_Model as tr_mod

test_file = 'NCBItestset_corpus.txt'
# Testing the model on the NCBI Testing Dataset, and calculating the error function. 

# Extracting the Testing Dataset
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

# Encoding the test data
test_dataset = tr_mod.NERDataset(test_sentences, test_tags, tr_mod.word_encoder, tr_mod.tag_encoder)
test_dataloader = tr_mod.DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=lambda x: x)

# Setting the model in evaluation mode
tr_mod.model.eval()
# with torch.no_grad():

result_file = "Test_Report_File.txt"
# with open (result_file, "w") as res_file:
#   file.write()
# file.write ('\n)