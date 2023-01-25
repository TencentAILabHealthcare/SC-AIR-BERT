from vocab_IR import WordVocab

def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-k", "--kmer", required=True, type=int)
    args = parser.parse_args()
   
    # kmer
    import itertools
    if(args.kmer!=1):
        amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', \
                'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        temp = list(itertools.product(amino_acids, repeat=args.kmer))
        sequence_characters = []
        if(args.kmer==3):
            for i in temp:
                sequence_characters.append(i[0]+i[1]+i[2])
        elif(args.kmer==2):
            for i in temp:
                sequence_characters.append(i[0]+i[1])
        sequence_characters = tuple(sequence_characters)
        vocab = WordVocab(sequence_characters=sequence_characters)
    else:
        vocab = WordVocab()
    print(vocab.stoi)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)

if __name__ == '__main__':
    build()