from nlstruct.datasets.base import NERDataset
from datasets import load_dataset

def tags_to_entities(words, ner_tags, tag_map):
    #this function takes a list of words and a list of tags and returns a list of entities
    #the tags are 0 for O, and 1 for type '1' and 2 for type '2' and so on
    #each entity is a dictionary with the following keys: 'id', 'type', 'begin', 'end', 'text'
    ann = []
    i=0
    while i < len(ner_tags):
        tag = ner_tags[i]
        if tag != 0:
            j = i+1
            while j < len(ner_tags) and ner_tags[j] == tag:
                j += 1
            ent_type = tag_map[tag]
            ent_id = f"T{len(ann)+1}"
            ent_text = ' '.join(words[i:j])
            ent_begin = len(' '.join(words[:i])) + 1 if i > 0 else 0
            ent_end = ent_begin + len(ent_text)
            ann.append({
                'entity_id': ent_id,
                'label': ent_type,
                'fragments': [{
                    'begin': ent_begin,
                    'end': ent_end
                }],
                'text': ent_text
            })
            i = j
        else:
            i += 1
    if "minéralogique" in words:
        print(ann)
        print(ner_tags)
    return ann

    
def load_from_hf(dataset, tag_map):
    examples = []
    # Load a brat dataset into a Dataset object
    for e in dataset:
        examples.append({
            'doc_id': e['docid'],
            'text': ' '.join(e['words']),
            'entities': tags_to_entities(e['words'], e['ner_tags'], tag_map),
        })
    return examples

class HuggingfaceNERDataset(NERDataset):
    def __init__(self, dataset_name: str, subset: str, tag_map: dict, preprocess_fn=None):
        train_data, val_data, test_data = self.extract(dataset_name, subset, tag_map)
        super().__init__(train_data, val_data, test_data, preprocess_fn=preprocess_fn)
    
    def extract(self, dataset_name, subset, tag_map):
        try :
            self.dataset = load_dataset(dataset_name, subset)
        except ValueError:
            raise ValueError(f"Dataset {dataset_name} does not exist. Please check the name of the dataset.")
        train_data = load_from_hf(self.dataset["train"], tag_map)
        test_data = load_from_hf(self.dataset["test"], tag_map)
        val_data = load_from_hf([], tag_map)
        return train_data, val_data, test_data
        
    

