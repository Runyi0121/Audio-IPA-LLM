from datasets import load_dataset, load_metric, Audio, concatenate_datasets, Dataset
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer, AdamW, get_scheduler,get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import json
import torch
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import random
from argparse import ArgumentParser
import pandas as pd
import os
import multiprocessing
from tqdm.auto import tqdm
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

#from add_forvo import add_language
#from preprocessor import Preprocessors, filter_low_quality, downsampling

def extract_all_chars_ipa(batch):
    # Change this function later at some point to create vocabulary based on
    # phonemes, not on characters
    all_text = " ".join(batch["ipa"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

def prepare_dataset_ipa(batch):
    audio = batch["audio"]

    # batched output is unbatched
    batch["input_values"] = processor(audio["array"],
    sampling_rate=audio["sampling_rate"]).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["ipa"]).input_ids
    return batch

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Split inputs and labels since they have to be of different lengths
        # and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
            )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
                )

        # Replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch

def remove_long_data(dataset, max_seconds=6):
    #convert pyarrow table to pandas
    dftest = dataset.to_pandas()
    #find out length of input_values
    dftest['len'] = dftest['input_values'].apply(len)
    #for wav2vec training we already resampled to 16khz
    #remove data that is longer than max_seconds (6 seconds ideal)
    maxLength = max_seconds * 16000 
    dftest = dftest[dftest['len'] < maxLength]
    dftest = dftest.drop('len', 1)
    #convert back to pyarrow table to use in trainer
    dataset = dataset.from_pandas(dftest)
    #directly remove do not wait for gc
    del dftest
    return dataset

def concatenate_common_voice(datasetlist: list):
    """
    Concatenate more than one datasets from Common Voice.
    Also consider using datasets.interleave_datasets(datasets: List[DatasetType]
    so that the new dataset is constructed by cycling between each source to get the examples.
    """
    init_data = datasetlist[0]
    for d in datasetlist:
        assert d.features.type == init_data.features.type
    concatenated = concatenate_datasets(datasetlist)
    return concatenated

def remove_space(batch: dict) -> dict:
    ipa = batch["ipa"]
    ipa = ipa.split()
    ipa = "".join(ipa)
    batch["ipa"] = ipa
    return batch

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-t", "--train_data", type=str,
                        help="Specify the file path of the training data.")
    parser.add_argument("-v", "--valid_data", type=str,
                        help="Specify the file path of the validation data.")
    parser.add_argument("-ts", "--train_samples", type=int,
                        help="Specify the number of samples for the training dataset.")
    parser.add_argument("-vs", "--valid_samples", type=int,
                        help="Specify the number of samples for the validation dataset.")
    parser.add_argument("-ns", "--no_space", action="store_true",
                        help="Use this flag if you want to omit spaces from the data.")
    parser.add_argument("-s", "--suffix", type=str, default="",
                        help="You can specify a suffix that will be added to the checkpoint directory to identify a trained model.")
    parser.add_argument("-vo", "--vocab_file", type=str,
                        help="Specify the vocab file name.")
    parser.add_argument("-e", "--num_epochs", type=int, default=30,
                        help="Specify the number of epochs during training. By default it is set to 30.")
    parser.add_argument("-np", "--num_proc", type=int, default=multiprocessing.cpu_count(),
                        help="Specify the number of CPUs for parallel processing. By default it uses maximum CPUs available.")
    args = parser.parse_args()
    return args

def load_data() -> Dataset:
    """Write a data-loading function.
    At this point it is not clear yet as to what the dataset will look like.
    Both CSV and TSV are handy formats for converting to the datasets.Dataset type,
    which we will need for handing it to Trainer later.
    The data should look like at least having two columns:
    1. The path to the audio file
    2. The IPA transcription of the audio file."""

    dataset = load_dataset("audiofolder", data_dir="/afs/crc.nd.edu/user/r/rshi2/mimik/xlsr-53-english/audios/", split= "train")
    dataset = dataset.rename_column("transcription", "ipa")
    #dataset = dataset.train_test_split(test_size=0.2, shuffle=True)
    
    # 60% train, 20% test + 20% validation
    train_testvalid = dataset.train_test_split(test_size=0.4, shuffle=True)
    # Split the 40% test + valid in half test, half valid
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5, shuffle=True)
    dataset = dict({
        'train': train_testvalid['train'],
        'test': test_valid['test'],
        'valid': test_valid['train']})

    return dataset

def generate_vocab(train, valid, test, vocab_file) -> None:
    print("Creating a character-level vocabulary...")
    vocab_train = train.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1, # Single batch
        keep_in_memory=True, # Keep the dataset in memory instead of writing it to a cache file.
        remove_columns = train.column_names
    )
    vocab_valid = valid.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=valid.column_names
    )
    vocab_test = test.map(
        extract_all_chars_ipa,
        batched=True,
        batch_size=-1,
        keep_in_memory=True,
        remove_columns=test.column_names
    )
    vocab_list = list(
        set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]) | set(vocab_valid["vocab"][0])
    )
    # add multiletter IPAs and other IPAs; only if necessary.
    # with open("full_vocab_ipa.txt", "r") as f:
    #     lines = f.readlines()
    #     ipa_all = set([l.strip() for l in lines])
    #     vocab_list = set(vocab_list) | ipa_all
    vocab_list = set(vocab_list)
    vocab_list = list(vocab_list)
    vocab_dict = {v: k for k, v in enumerate(vocab_list)}
    print("Vocab created. Details:")
    print("vocab_dict_ipa: {}".format(len(vocab_dict)))

    # Preprocessing necessary for CTC
    # Add [UNK], [PAD]
    print("Adding [UNK] and [PAD]...")
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    print("[UNK] and [PAD] added")

    print("Writing vocab json files...")
    with open(vocab_file, 'w') as f:
        json.dump(vocab_dict, f)
    print("Vocab json files created")

if __name__ == "__main__":
    
    #assert torch.cuda.is_available() , print("The program has not found any CUDA devices")
    
    args = get_args()
    
    # Data loading
    dataset = load_data()
    train = dataset["train"]
    valid = dataset["valid"]
    test  = dataset["test"]
    
    print(dataset)
    # Remove spaces if specified
    train = train.map(remove_space)
    valid = valid.map(remove_space)
    test = test.map(remove_space)
    assert " " not in train["ipa"], print("Space removal did not work correctly for the train set.")
    assert " " not in valid["ipa"], print("Space removal did not work correctly for the valid set.")
    assert " " not in test["ipa"], print("Space removal did not work correctly for the test set.")
            
    # You can also shuffle the dataset here using the Dataset.shuffle(seed=) function, if necessary.

    train = train.cast_column("audio", Audio(sampling_rate = 16000))
    valid = valid.cast_column("audio", Audio(sampling_rate = 16000))
    test = test.cast_column("audio", Audio(sampling_rate = 16000))

    # Specify the folder where you want to save the dataset
    download_folder = os.path.dirname("./dataset_storage")

    # Save the dataset to the specified folder
    valid.save_to_disk(download_folder)

    # Vocab
    generate_vocab(train, valid, test, args.vocab_file)

    # Create Tokenizers
    print("Creating Tokenizers...")
    # Be careful to load the correct vocab file.
    tokenizer = Wav2Vec2CTCTokenizer("./{}".format(args.vocab_file), unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
    print("Tokenizers created") 

    # Create a Feature Extractor
    # You might need to change the sampling rate, depending on the original dataset.
    print("Creating Feature Extractor...")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)
    print("Feature Extractor created") 

    # Define Processors
    print("creating Processors...")
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    print("Processors created") 

    # Set the sampling rate to 16,000Hz, if necessary.
    # print("Adjusting the sampling rate to 16,000Hz...")
    # common_voice_train = common_voice_train.cast_column("audio", Audio(sampling_rate=16_000))
    # common_voice_test = common_voice_test.cast_column("audio", Audio(sampling_rate=16_000))
    # print("Sampling rate adjustment done")

    print("Preprocessing the dataset...")
    # Try removing `num_proc=` if you encounter any errors while running this part
    train = train.map(
        prepare_dataset_ipa,
        num_proc=args.num_proc
    )
    valid = valid.map(
        prepare_dataset_ipa,
        num_proc=args.num_proc
    )
    test = test.map(
        prepare_dataset_ipa,
        num_proc=args.num_proc
    )

    # Clip the audio length if you get memory crash during training
    # print("Removing audio files longer than 6 secs...")
    # train = remove_long_data(train)
    # valid = remove_long_data(valid)
    # print("Dataset lengths to be trained and tested:")
    print("Train size:", len(train))
    print("Valid size:", len(valid))
    print("Test size:", len(test))
    print("Preprocessing done")

    print("Creating the data collator")
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    print("Data collator created")
    
    # Model
    print("Defining the model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
        )
    print("Model defined")

    # Freeze the feature extractor so that it won't be changed by the fine-tuning
    print("Freezing the feature extractor...") 
    model.freeze_feature_extractor()
    print("Feature extractor frozen")

    output_dir = "./wav2vec2-large-xlsr-53-english"
    if args.suffix:
        output_dir += args.suffix


    # Training

    optimizer = AdamW(model.parameters(), lr=3e-4)
    num_training_steps = args.num_epochs * len(train)
    lr_scheduler = get_cosine_with_hard_restarts_schedule_with_warmup ( optimizer=optimizer, num_training_steps = num_training_steps, num_warmup_steps=50 )

    # Training
    print("Beginning the training...") 
    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=args.num_epochs,
        fp16=False,
        save_steps=300,
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=0,
        save_total_limit=2,
    )

    #initialize the trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=processor.feature_extractor,
        optimizers = [optimizer, lr_scheduler]
        )

    trainer.train()
    trainer.evaluate()
    trainer.save_state()
    trainer.save_model()

