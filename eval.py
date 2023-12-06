import collections
from collections import abc
from evaluate import load
from datasets import load_dataset, Audio, Dataset, IterableDataset
from transformers import pipeline, AutoProcessor, AutoModelForCTC
from argparse import ArgumentParser
import torch
import time
import tqdm
import datasets
from typing import Tuple
from multiprocessing import Pool
import os
from torchmetrics.text import WordInfoLost
import string
from functools import partial
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer
import argparse
import main
from datasets import load_from_disk

#special_vocab.json
#tokenizer file incorporated

def get_args():
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="./",
                        help="Specify the model path.")
    parser.add_argument("-o", "--output", type=str, default=None,
                        help="Specify the stats file name.")
    parser.add_argument("--dev", action="store_true",
                        help="Use this flag to evaluate on the dev set.")
    parser.add_argument("-s", "--sample", type=int, default=100)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_space", action="store_true")
    args = parser.parse_args()

    # assert args.language == os.path.basename(args.model)
    if args.output == None:
        if args.dev:
            args.output = os.path.join(args.model, "eval_dev.txt")
        else:
            args.output = os.path.join(args.model, "eval.txt")
    return args

def load_test(lang, stream=True):
    """Load the test data."""
    download_mode = "force_redownload"
    if stream:
        if args.dev:
            datasplit = "validation"
        else:
            datasplit = "test"
        test = load_dataset("",
                            split=datasplit,
                            streaming=True)
        test = test.take(args.sample)
    else:    
        test = load_dataset("",
                            split=datasplit,
                            download_mode=download_mode)

    def gen_from_iterable_dataset(iterable_dataset: IterableDataset):
        yield from iterable_dataset

    test = Dataset.from_generator(partial(gen_from_iterable_dataset, test),
                                  features=test.features)
    test = test.cast_column("audio", Audio(sampling_rate=16000))
    return test

def inference(batch):
    """Predict a transcription per sample."""
    inputs = processor(batch["audio"]["array"],
                       sampling_rate=16_000,
                       return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)

    reference = batch["ipa"]
    prediction = transcription[0]
    return reference, prediction

def normalize(text: str):
    """All lowercase and without punctuation"""
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

def eval_metrics(predictions: list, references: list) -> dict:
    """Calculate metrics.
    
    Returns: a dict of scores containing CER, WER, and WIL,
    both for raw sentences and normalized sentences.
    """
    cer_score = cer.compute(predictions=predictions, references=references)
    #wer_score = wer.compute(predictions=predictions, references=references)
    #wil_score = wil(predictions, references)
    #wil_score = wil_score.item()

    predictions_norm = [normalize(p) for p in predictions]
    references_norm = [normalize(r) for r in references]
    cer_score_norm = cer.compute(predictions=predictions_norm, references=references_norm)
    #wer_score_norm = wer.compute(predictions=predictions_norm, references=references_norm)
    #wil_score_norm = wil(predictions_norm, references_norm)

    results = {"CER": cer_score,
               "CERnorm": cer_score_norm,
               }
    return results

if __name__ == "__main__":
    args = get_args()
        
    cer = load("cer")
    #wer = load("wer")
    #wil = WordInfoLost()
        
    print("Dataset loaded")
    if args.dev:
        eval_set_name = "Valid"
    else:
        eval_set_name = "Test"

    test = load_from_disk("./")

    eval_num_sample = len(test)

    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForCTC.from_pretrained(args.model)
    model.to(torch.float64)
    print("dtype:", model.dtype)

    # calculate CER and WER
    start = time.time()
    predictions = []
    references = []
    with Pool(4) as p:
        # Multiprocessing with 4 cores; this must be in the main part
        for reference, prediction in p.map(inference, test):
            predictions.append(prediction)
            references.append(reference)
    assert len(references) == len(test), print("Ref:", len(references), "Pred:", len(predictions), "Test:", len(test))
    assert len(references) == len(predictions)
    
    results = eval_metrics(predictions, references)
    end = time.time()
    total_time = end - start

    with open(args.output, "w") as f:
        f.write("{} set size: {}\n".format(eval_set_name, eval_num_sample))
        f.write("CER: {}\n".format(results["CER"]))
        f.write("CER_norm: {}\n".format(results["CERnorm"]))
        f.write("Total time:{}\n".format(total_time))
        f.write("Reference\tPrediction\n")
        for i in range(len(predictions)):
            f.write("\n{}\n{}\n".format(references[i], predictions[i]))
            