from transformers import LineByLineTextDataset

class Dataset(LineByLineTextDataset):
    def __init__(self, tokenizer, file_path, block_size, label_file_path,class_name):
        super().__init__(tokenizer=tokenizer, file_path=file_path, block_size=block_size)
        self.label_file_path = label_file_path
        self.class_name = class_name

    def __getitem__(self, idx):
        text = super().__getitem__(idx)
        label = 0
        ID = ""
        with open(self.label_file_path, encoding="utf-8") as f:
            line = f.readlines()[idx].strip().split("\t")
            if(line[1]==self.class_name):
                label = 1
            else:
                label = 0
            ID = line[0]
        return {"text": text, "labels": label, "ID":int(ID)}
    