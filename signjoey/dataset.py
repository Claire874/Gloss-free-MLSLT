# coding: utf-8
"""
Data module
"""
import h5py
import torch
from torchtext import data
from torchtext.data import Field, RawField
from typing import List, Tuple
import pickle
import gzip
import torch
import numpy as np
import re
from zhon.hanzi import punctuation

def load_dataset_file(filename):
    load_object= np.load(filename, allow_pickle=True)
    del load_object.item()['prefix']
    #load_object = h5py.File(filename,'r')
    return load_object.item()

# feature: load_object[*][:]
def load_sign_feature(feature_path):
    feature = h5py.File(feature_path,'r')
    return feature

def _load_text_data(filename: str) -> List[str]:
    with open(filename, 'r', encoding='UTF-8') as f:
        return [line.strip() for line in f]

class SignTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.sgn), len(ex.txt))

    def __init__(
        self,
        path: str,
        fields: Tuple[RawField, RawField, Field, Field, Field],
        **kwargs
    ):
        """Create a SignTranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [
                ("sequence", fields[0]),
                ("signer", fields[1]),
                ("sgn", fields[2]),
                ("gls", fields[3]),
                ("txt", fields[4]),
            ]

        if not isinstance(path, list):
            path = [path]

        samples = {}
        #text_data_back = _load_text_data('./csl_backtranslation.txt')
        text_index = 0
        samples_corr_back = {}
        samples_smkd_back = {}


        for annotation_file in path:
            tmp = load_dataset_file(annotation_file)
            for s in tmp:
                seq_id = tmp[s]["fileid"]
                if 'train' in annotation_file:
                    #feature_path_skmd = "./data/SKMD-CSL-Daily/train.h5"
                    #feature_path_smkd = "./signjoey/features_csl_gloss_free/train/" + seq_id +"_visual_features.npy"
                    #feature_path_corr = "./signjoey/features-gloss-free-corr-csl/train/" + seq_id + "_Corr_visual_features.npy"
                    feature_path_slowfast = "./signjoey/features_csl_slowfast/train/" + seq_id + "_features.npy"
                elif 'dev' in annotation_file:
                    #feature_path_smkd = "./data/SKMD-CSL-Daily/dev.h5"
                    #feature_path_smkd = "./signjoey/features_csl_gloss_free/dev/" + seq_id +"_visual_features.npy"
                    #feature_path_corr = "./signjoey/features-gloss-free-corr-csl/dev/" + seq_id + "_Corr_visual_features.npy"
                    feature_path_slowfast = "./signjoey/features_csl_slowfast/dev/" + seq_id + "_features.npy"
                else:
                    #feature_path_skmd = "./data/SKMD-CSL-Daily/test.h5"
                    #feature_path_smkd = "./signjoey/features_csl_gloss_free/test/" + seq_id +"_visual_features.npy"
                    #feature_path_corr = "./signjoey/features-gloss-free-corr-csl/test/" + seq_id + "_Corr_visual_features.npy"
                    feature_path_slowfast = "./signjoey/features_csl_slowfast/test/" + seq_id + "_features.npy"

                #feature_from_skmd = load_sign_feature(feature_path_skmd)[str(s)][:]
                #feature_from_smkd = np.load(feature_path_smkd, allow_pickle=True).item()
                #feature_from_corr = np.load(feature_path_corr, allow_pickle=True).item()
                feature_from_slowfast = np.load(feature_path_slowfast, allow_pickle=True).item()


                #new_sgn_skmd = torch.tensor(feature_from_skmd)
                
                #new_sgn_smkd = feature_from_smkd["features"]
                #new_sgn_corr = feature_from_corr["features"]
                new_sgn_slowfast = feature_from_slowfast["features"]

                if seq_id in samples:
                #feature may load here
                    assert samples[seq_id]["name"] == s["name"]
                    assert samples[seq_id]["signer"] == s["signer"]
                    assert samples[seq_id]["gloss"] == s["gloss"]
                    assert samples[seq_id]["text"] == s["text"]
                    samples[seq_id]["sign"] = torch.cat(
                        [samples[seq_id]["sign"], s["sign"]], axis=1
                    )
                else:
                    #gloss = ''.join(i + ' ' for i in tmp[s]['original_info'][0]['label_gloss'])
                    #text = ''.join(i + ' ' for i in tmp[s]['original_info'][0]['label_char'])
                    gloss = ''.join(i for i in tmp[s]['original_info'][0]['label_gloss'])
                    text = ''.join(i for i in tmp[s]['original_info'][0]['label_char'])
                    #text_word = ''.join(i + ' ' for i in tmp[s]['original_info'][0]['label_word'])
                    #for i in punctuation:
                    #    text = text.replace(i,"")
                    #text_line_check = text_data_back[text_index] if text_index < len(text_data_back) else ""
                    #text_index += 1
                    samples[seq_id] = {
                        "name": tmp[s]["fileid"],
                        "signer": tmp[s]["signer"],
                        "gloss": text,
                        "text": text,
                        "sign": new_sgn_slowfast,
                        #"sign": s["sign"],
                        #change???   
                    }
                #if 'train' in annotation_file:
                #    text_line_back = text_data_back[text_index] if text_index < len(text_data_back) else ""
                #    text_index += 1
                #    samples_smkd_back[seq_id + "f3t3"] = {
                #    "name": tmp[s]["fileid"],
                #    "signer": tmp[s]["signer"],
                #    "gloss": gloss,
                    #    "text": s["text"],
                #    "text": text_line_back,
                #    "sign": new_sgn_smkd,
                #    }
                    
                #    samples_corr_back[seq_id + "f2t2"] = {
                #    "name": tmp[s]["fileid"],
                #    "signer": tmp[s]["signer"],
                #    "gloss": gloss,
                    #    "text": s["text"],
                #    "text": text_line_back,
                #    "sign": new_sgn_corr,
                #    }
                #print('ori:', text,'back',text_line_back)

        examples = []
        #samples.update(samples_corr_back)
        #samples.update(samples_smkd_back)
       
        for s in samples:                    	
            sample = samples[s]
            examples.append(
                data.Example.fromlist(
                    [
                        sample["name"],
                        sample["signer"],
                        # This is for numerical stability
                        sample["sign"] + 1e-8,
                        sample["gloss"].strip(),
                        sample["text"].strip(),
                    ],
                    fields,
                )
            )
        super().__init__(examples, fields, **kwargs)
