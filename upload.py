from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import argparse

def upload(model_path, vocab_file, directory):
    """
    Running this function will create additional necessary files such as `special_tokens_map.json`
    and make it ready for the model to be uploaded on the huggingface repository.
    """
    
    print("Reading the model...")
    model = Wav2Vec2ForCTC.from_pretrained(model_path)
    print("Model read")

    print("Reading the tokenizer...")
    if not vocab_file.startswith("/"):
        # relative path
        vocab_file = "./" + vocab_file
    tokenizer_sentence = Wav2Vec2CTCTokenizer(vocab_file,
                                              unk_token="[UNK]",
                                              pad_token="[PAD]",
                                              word_delimiter_token="|")
    print("Tokenizer read")

    print("Saving the model...")
    model.save_pretrained(directory)
    print("Model saved")
    print("Saving the tokenizer...")
    tokenizer_sentence.save_pretrained(directory)
    print("Tokenizer saved")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path", default="./", type=str,
        help="Model (checkpoint) directory path")
    parser.add_argument("-v", "--vocab_file", required=True, type=str,
        help="Vocab file path")
    parser.add_argument("-d", "--directory", default="./", type=str,
        help="Destination of the saved model")
    args = parser.parse_args()
    model_path = args.model_path
    vocab_file = args.vocab_file
    directory = args.directory
    upload(model_path, vocab_file, directory)