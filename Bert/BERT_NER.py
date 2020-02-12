#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[3]:


path='BC5CDR-disease'
os.listdir(path)


# In[4]:


import tensorflow as tf

import pandas as pd
import numpy as np  
import re
import pickle

import keras as keras
from keras.models import load_model
from keras import backend as K
from keras import Input, Model
from keras import optimizers

import codecs
from tqdm import tqdm
import shutil


# In[5]:


import warnings
import tensorflow as tf
warnings.filterwarnings(action='ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.ERROR)


# In[6]:


#!pip install keras-bert
#!pip install keras-radam


# In[7]:


from keras_bert import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import Tokenizer
from keras_bert import AdamWarmup, calc_train_steps

from keras_radam import RAdam


# In[8]:


def _read_data(input_file):
  with open(input_file) as f:
    lines = []
    words = []
    labels = []
    for line in f:
      contends = line.strip()
      if len(contends) == 0:
        assert len(words) == len(labels)
        if len(words) >30:
          while len(words) > 30:
            tlabel = labels[:30]
            for tidx in range(len(tlabel)):
              if tlabel.pop() == 'O':
                break
            l = ' '.join([label for label in labels[:len(tlabel)+1] if len(label) > 0])
            w = ' '.join([word for word in words[:len(tlabel)+1] if len(word) > 0])
            lines.append([l,w])
            words = words[len(tlabel)+1:]
            labels = labels[len(tlabel)+1:]
        if len(words) == 0:
          continue
        l = ' '.join([label for label in labels if len(label) > 0])
        w = ' '.join([word for word in words if len(word) > 0])
        lines.append([l, w])
        words = []
        labels = []
        continue
      word = line.strip().split()[0]
      label = line.strip().split()[-1]
      words.append(word)
      labels.append(label)
    return lines        


# In[9]:


dataset_path = "BC5CDR-disease"

train_set=_read_data(dataset_path+"/train.tsv")
valid_set=_read_data(dataset_path+"/devel.tsv")
test_set =_read_data(dataset_path+"/test.tsv")


# In[10]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re
import unicodedata
import six
import tensorflow as tf


def validate_case_matches_checkpoint(do_lower_case, init_checkpoint):
  """Checks whether the casing config is consistent with the checkpoint name."""

  if not init_checkpoint:
    return

  m = re.match("^.*?([A-Za-z0-9_-]+)/bert_model.ckpt", init_checkpoint)
  if m is None:
    return

  model_name = m.group(1)

  lower_models = [
      "uncased_L-24_H-1024_A-16", "uncased_L-12_H-768_A-12",
      "multilingual_L-12_H-768_A-12", "chinese_L-12_H-768_A-12"
  ]

  cased_models = [
      "cased_L-12_H-768_A-12", "cased_L-24_H-1024_A-16",
      "multi_cased_L-12_H-768_A-12"
  ]

  is_bad_config = False
  if model_name in lower_models and not do_lower_case:
    is_bad_config = True
    actual_flag = "False"
    case_name = "lowercased"
    opposite_flag = "True"

  if model_name in cased_models and do_lower_case:
    is_bad_config = True
    actual_flag = "True"
    case_name = "cased"
    opposite_flag = "False"

  if is_bad_config:
    raise ValueError(
        "You passed in `--do_lower_case=%s` with `--init_checkpoint=%s`. "
        "However, `%s` seems to be a %s model, so you "
        "should pass in `--do_lower_case=%s` so that the fine-tuning matches "
        "how the model was pre-training. If this error is wrong, please "
        "just comment out this check." % (actual_flag, init_checkpoint,
                                          model_name, case_name, opposite_flag))


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def printable_text(text):
  """Returns text encoded in a way suitable for print or `tf.logging`."""

  # These functions want `str` for both Python2 and Python3, but in one case
  # it's a Unicode string and in the other it's a byte string.
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text
    elif isinstance(text, unicode):
      return text.encode("utf-8")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with tf.gfile.GFile(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


def convert_by_vocab(vocab, items):
  """Converts a sequence of [tokens|ids] using the vocab."""
  output = []
  for item in items:
    output.append(vocab[item])
  return output


def convert_tokens_to_ids(vocab, tokens):
  return convert_by_vocab(vocab, tokens)


def convert_ids_to_tokens(inv_vocab, ids):
  return convert_by_vocab(inv_vocab, ids)


def whitespace_tokenize(text):
  """Runs basic whitespace cleaning and splitting on a piece of text."""
  text = text.strip()
  if not text:
    return []
  tokens = text.split()
  return tokens


class FullTokenizer(object):
  """Runs end-to-end tokenziation."""

  def __init__(self, vocab_file, do_lower_case=True):
    self.vocab = load_vocab(vocab_file)
    self.inv_vocab = {v: k for k, v in self.vocab.items()}
    self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case)
    self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

  def tokenize(self, text):
    split_tokens = []
    for token in self.basic_tokenizer.tokenize(text):
      for sub_token in self.wordpiece_tokenizer.tokenize(token):
        split_tokens.append(sub_token)

    return split_tokens

  def convert_tokens_to_ids(self, tokens):
    return convert_by_vocab(self.vocab, tokens)

  def convert_ids_to_tokens(self, ids):
    return convert_by_vocab(self.inv_vocab, ids)


class BasicTokenizer(object):
  """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

  def __init__(self, do_lower_case=True):
    """Constructs a BasicTokenizer.

    Args:
      do_lower_case: Whether to lower case the input.
    """
    self.do_lower_case = do_lower_case

  def tokenize(self, text):
    """Tokenizes a piece of text."""
    text = convert_to_unicode(text)
    text = self._clean_text(text)

    # This was added on November 1st, 2018 for the multilingual and Chinese
    # models. This is also applied to the English models now, but it doesn't
    # matter since the English models were not trained on any Chinese data
    # and generally don't have any Chinese data in them (there are Chinese
    # characters in the vocabulary because Wikipedia does have some Chinese
    # words in the English Wikipedia.).
    text = self._tokenize_chinese_chars(text)

    orig_tokens = whitespace_tokenize(text)
    split_tokens = []
    for token in orig_tokens:
      if self.do_lower_case:
        token = token.lower()
        token = self._run_strip_accents(token)
      split_tokens.extend(self._run_split_on_punc(token))

    output_tokens = whitespace_tokenize(" ".join(split_tokens))
    return output_tokens

  def _run_strip_accents(self, text):
    """Strips accents from a piece of text."""
    text = unicodedata.normalize("NFD", text)
    output = []
    for char in text:
      cat = unicodedata.category(char)
      if cat == "Mn":
        continue
      output.append(char)
    return "".join(output)

  def _run_split_on_punc(self, text):
    """Splits punctuation on a piece of text."""
    chars = list(text)
    i = 0
    start_new_word = True
    output = []
    while i < len(chars):
      char = chars[i]
      if _is_punctuation(char):
        output.append([char])
        start_new_word = True
      else:
        if start_new_word:
          output.append([])
        start_new_word = False
        output[-1].append(char)
      i += 1

    return ["".join(x) for x in output]

  def _tokenize_chinese_chars(self, text):
    """Adds whitespace around any CJK character."""
    output = []
    for char in text:
      cp = ord(char)
      if self._is_chinese_char(cp):
        output.append(" ")
        output.append(char)
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)

  def _is_chinese_char(self, cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
      return True

    return False

  def _clean_text(self, text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
      cp = ord(char)
      if cp == 0 or cp == 0xfffd or _is_control(char):
        continue
      if _is_whitespace(char):
        output.append(" ")
      else:
        output.append(char)
    return "".join(output)


class WordpieceTokenizer(object):
  """Runs WordPiece tokenziation."""

  def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=200):
    self.vocab = vocab
    self.unk_token = unk_token
    self.max_input_chars_per_word = max_input_chars_per_word

  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

    text = convert_to_unicode(text)

    output_tokens = []
    for token in whitespace_tokenize(text):
      chars = list(token)
      if len(chars) > self.max_input_chars_per_word:
        output_tokens.append(self.unk_token)
        continue

      is_bad = False
      start = 0
      sub_tokens = []
      while start < len(chars):
        end = len(chars)
        cur_substr = None
        while start < end:
          substr = "".join(chars[start:end])
          if start > 0:
            substr = "##" + substr
          if substr in self.vocab:
            cur_substr = substr
            break
          end -= 1
        if cur_substr is None:
          is_bad = True
          break
        sub_tokens.append(cur_substr)
        start = end

      if is_bad:
        output_tokens.append(self.unk_token)
      else:
        output_tokens.extend(sub_tokens)
    return output_tokens


def _is_whitespace(char):
  """Checks whether `chars` is a whitespace character."""
  # \t, \n, and \r are technically contorl characters but we treat them
  # as whitespace since they are generally considered as such.
  if char == " " or char == "\t" or char == "\n" or char == "\r":
    return True
  cat = unicodedata.category(char)
  if cat == "Zs":
    return True
  return False


def _is_control(char):
  """Checks whether `chars` is a control character."""
  # These are technically control characters but we count them as whitespace
  # characters.
  if char == "\t" or char == "\n" or char == "\r":
    return False
  cat = unicodedata.category(char)
  if cat.startswith("C"):
    return True
  return False


def _is_punctuation(char):
  """Checks whether `chars` is a punctuation character."""
  cp = ord(char)
  # We treat all non-letter/number ASCII as punctuation.
  # Characters such as "^", "$", and "`" are not in the Unicode
  # Punctuation class but we treat them as punctuation anyways, for
  # consistency.
  if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
      (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
    return True
  cat = unicodedata.category(char)
  if cat.startswith("P"):
    return True
  return False


# In[11]:


#import os
#from google.colab import drive
#drive.mount('/content/gdrive/')

path='models'


# In[12]:


def InputFeatures(input_ids, segment_ids, label_ids):
    input_ids = input_ids
    segment_ids = segment_ids
    label_ids = label_ids

    return input_ids, segment_ids, label_ids



# In[13]:


def convert_single_example(ex_index, example, label_list, max_seq_length,tokenizer): 
    #vocab=path+'/vocab.txt' #구드path로 변경!
   # wordtt= FullTokenizer(vocab) #class형태로 선언

    label_map = {}
    for (i, label) in enumerate(label_list,1):
        label_map[label] = i

   # print("label_map: ", label_map) --> label_map:  {'B': 1, 'I': 2, 'O': 3, 'X': 4, '[CLS]': 5, '[SEP]': 6}
   # with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'),'wb') as w:
   #    pickle.dump(label_map,w)
    textlist = example[1].split(' ') #example.text.split(' ')을 index형태로 바꿈
    labellist = example[0].split(' ') #example.label.split(' ')을 index형태로 바꿈
    tokens = []
    labels = []
    for i, word in enumerate(textlist):
        token = tokenizer.tokenize(word) #사용!! 
        tokens.extend(token)
        label_1 = labellist[i]
        for m in range(len(token)):
            if m==0:
                labels.append(label_1)
            else:
                labels.append("X")
    if len(tokens) >= max_seq_length - 1:
        tokens = tokens[0:(max_seq_length - 2)]
        labels = labels[0:(max_seq_length - 2)]
    ntokens = []
    segment_ids = []
    label_ids = []
    ntokens.append("[CLS]")
    segment_ids.append(0)
    # append("O") or append("[CLS]") not sure!
    label_ids.append(label_map["[CLS]"])
    for i, token in enumerate(tokens):
        ntokens.append(token)
        segment_ids.append(0)
        label_ids.append(label_map[labels[i]])
    ntokens.append("[SEP]")
    segment_ids.append(0)
    # append("O") or append("[SEP]") not sure!
    label_ids.append(label_map["[SEP]"])
    input_ids = tokenizer.convert_tokens_to_ids(ntokens)
    input_mask = [1] * len(input_ids)
    #label_mask = [1] * len(input_ids)
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        # we don't concerned about it!
        label_ids.append(0)
        ntokens.append("**NULL**")
        #label_mask.append(0)
    # print(len(input_ids))
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    #assert len(label_mask) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        #tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))
        #tf.logging.info("label_mask: %s" % " ".join([str(x) for x in label_mask]))

    #print(labels)
    input_ids, segment_ids, label_ids = InputFeatures( #함수형태로 전환했기때문에 받는 변수도 3개로 변경
        input_ids=input_ids,
        segment_ids=segment_ids,
        label_ids=label_ids,
        #label_mask = label_mask
    )
    #write_tokens(ntokens)
    return input_ids, segment_ids, label_ids

            


# In[14]:


def data_loader(data_set):
    vocab=path+'/vocab.txt'
    label_list=['B','I','O','X','[CLS]','[SEP]'] #101: [CLS], 102: [SEP]
    max_seq_length= 128
    tokenizer= FullTokenizer(vocab)

    result_input_ids=[]
    result_seg_ids=[]
    y=[]

    for i, example in enumerate(data_set):
        input_ids, segment_ids, label_ids=convert_single_example(i , example, label_list,max_seq_length, tokenizer)
        result_input_ids.append(input_ids)
        result_seg_ids.append(segment_ids)

        y.append(label_ids)
        
    data_x = []
    data_y = []
    data_x.append(np.array(result_input_ids))
    data_x.append(np.array(result_seg_ids))

    data_y.append(np.array(y))
        
    return data_x, data_y


# In[15]:


train_x, train_y = data_loader(train_set)
valid_x, valid_y = data_loader(valid_set)
test_x, test_y = data_loader(test_set)


# In[16]:


print(np.array(train_x).shape)
print(np.array(valid_x).shape)
print(np.array(test_x).shape)
print(np.array(train_y).shape)
print(np.array(valid_y).shape)
print(np.array(test_y).shape)


# In[17]:


train_y # {'B': 1, 'I': 2, 'O': 3, 'X': 4, '[CLS]': 5, '[SEP]': 6}

from tensorflow.keras.utils import to_categorical
train_y_hot = to_categorical(train_y)
valid_y_hot = to_categorical(valid_y)


# In[18]:


#import tarfile
#tar = tarfile.open("models/biobert_v1.1_pubmed.tar")
#tar.extractall()
#tar.close()


# In[19]:


SEQ_LEN = 128 #문장의 길이 최대 128 -> 너무 길면 메모리가...
BATCH_SIZE = 16 #bert는 모델이 무거움(fine tuning 할 때는 작게 잡음, para들이 너무 많이 변하면 안됨..?)
EPOCHS=4
LR=1e-5 #fine tuning 할 때는 lr를 작게 함

pretrained_path ="models"
#pretrained_path="bert"

os.listdir(pretrained_path)


# In[20]:


config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'model.ckpt-1000000')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

layer_num = 12
model = load_trained_model_from_checkpoint( #사전학습된 모델 불러오기
    config_path,
    checkpoint_path,
    training= False,
    trainable=True,
    seq_len=SEQ_LEN)


# In[21]:


model.summary()


# In[22]:


def get_bert_finetuning_model(model):
    inputs = model.inputs[:2] #segment, token 두개의 input
    dense = model.output
    out = keras.layers.Dropout(0.2)(dense)
    output_layer=keras.layers.Dense(7, activation='softmax')(out)
    model = keras.models.Model(inputs, output_layer)
    model.compile(
        
      optimizer=RAdam(learning_rate=LR, weight_decay=0.001),
      loss="categorical_crossentropy",
      metrics=["categorical_accuracy"])

    return model


# In[ ]:


K.tensorflow_backend._get_available_gpus()

sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init = tf.variables_initializer([v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables])
sess.run(init)

bert_model = get_bert_finetuning_model(model)
bert_model.summary()
#validation_data=(valid_x, valid_y_hot[0]),
history = bert_model.fit(train_x, train_y_hot[0], epochs=EPOCHS, batch_size=10,  shuffle=False, verbose=1)


# In[2]:


#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


# In[ ]:




