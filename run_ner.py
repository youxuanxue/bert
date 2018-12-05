# !/usr/bin/python
# -*- coding: utf-8 -*-
import collections
import json
import os

import tensorflow as tf
import tensorflow.contrib.crf

import modeling
import optimization
import tokenization

__author__ = 'xuejiao'

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
  "bert_config_file", None,
  "The config json file corresponding to the pre-trained BERT model. "
  "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("label_vocab_file", None,
                    "The vocabulary file for the NER label.")

flags.DEFINE_string(
  "output_dir", None,
  "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string("train_file", None,
                    "SQuAD json for training. E.g., train-v1.1.json")

flags.DEFINE_string(
  "predict_file", None,
  "SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")

flags.DEFINE_string(
  "init_checkpoint", None,
  "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
  "do_lower_case", True,
  "Whether to lower case the input text. Should be True for uncased "
  "models and False for cased models.")

flags.DEFINE_integer(
  "max_seq_length", 384,
  "The maximum total input sequence length after WordPiece tokenization. "
  "Sequences longer than this will be truncated, and sequences shorter "
  "than this will be padded.")

flags.DEFINE_integer(
  "doc_stride", 128,
  "When splitting up a long document into chunks, how much stride to "
  "take between chunks.")

flags.DEFINE_integer(
  "max_query_length", 64,
  "The maximum number of tokens for the question. Questions longer than "
  "this will be truncated to this length.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("predict_batch_size", 8,
                     "Total batch size for predictions.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
  "warmup_proportion", 0.1,
  "Proportion of training to perform linear learning rate warmup for. "
  "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer(
  "n_best_size", 20,
  "The total number of n-best predictions to generate in the "
  "nbest_predictions.json output file.")

flags.DEFINE_integer(
  "max_answer_length", 30,
  "The maximum length of an answer that can be generated. This is needed "
  "because the start and end predictions are not conditioned on one another.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
  "tpu_name", None,
  "The Cloud TPU to use for training. This should be either the name "
  "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
  "url.")

tf.flags.DEFINE_string(
  "tpu_zone", None,
  "[Optional] GCE zone where the Cloud TPU is located in. If not "
  "specified, we will attempt to automatically detect the GCE project from "
  "metadata.")

tf.flags.DEFINE_string(
  "gcp_project", None,
  "[Optional] Project name for the Cloud TPU-enabled project. If not "
  "specified, we will attempt to automatically detect the GCE project from "
  "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
  "num_tpu_cores", 8,
  "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool(
  "verbose_logging", False,
  "If true, all of the warnings related to data processing will be printed. "
  "A number of warnings are expected for a normal SQuAD evaluation.")

flags.DEFINE_bool(
  "version_2_with_negative", False,
  "If true, the SQuAD examples contain some that do not have an answer.")

flags.DEFINE_float(
  "null_score_diff_threshold", 0.0,
  "If null_score - best_non_null is greater than the threshold predict null.")


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               tokens,
               input_ids,
               input_mask,
               segment_ids,
               labels,
               label_ids):
    self.tokens = tokens
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.labels = labels
    self.label_ids = label_ids


class FeatureWriter(object):
  """Writes InputFeature to TF example file."""

  def __init__(self, filename, is_training):
    self.filename = filename
    self.is_training = is_training
    self.num_features = 0
    self._writer = tf.python_io.TFRecordWriter(filename)

  def process_feature(self, feature):
    """Write a InputFeature to the TFRecordWriter as a tf.train.Example."""
    self.num_features += 1

    if self.num_features % 10000 == 0:
      tf.logging.info("process feature %s", self.num_features)

    def create_int_feature(values):
      feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
      return feature

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)

    if self.is_training:
      features["label_ids"] = create_int_feature(feature.label_ids)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    self._writer.write(tf_example.SerializeToString())

  def close(self):
    self._writer.close()


RawResult = collections.namedtuple("RawResult",
                                   ["input_ids", "predict_ids"])


def write_predictions(inv_vocab, all_results, output_prediction_file):
  tf.logging.info("Writing predictions to: %s" % (output_prediction_file))

  def build_result(result):
    task_result = []
    cur_token = []
    chars = [inv_vocab[x] for x in result.input_ids]
    preds = [inv_vocab[x] for x in result.predict_ids]
    for char, pred in zip(chars, preds):
      if pred in ["S", "O", "<UNK>", "E"]:
        cur_token.append(char)
        task_result.append("".join(cur_token))
        cur_token = []
      else:
        pieces = pred.split('-')
        if len(pieces) == 2:
          if pieces[0] in ["S", "O", "<UNK>", "E"]:
            cur_token.append(char + "/" + pieces[1])
            task_result.append("".join(cur_token))
            cur_token = []
          else:
            cur_token.append(char)
        else:
          cur_token.append(char)
    if cur_token:
      task_result.append("".join(cur_token))
    return "".join(task_result)

  final_results = []
  for result in all_results:
    final_results.append(build_result(result))

  with tf.gfile.GFile(output_prediction_file, "w") as writer:
    writer.write(json.dumps(final_results, indent=4) + "\n")


def parse_file_len(fname):
  i = -1
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1


def read_ner_features(file_name, text_tokenizer, label_tokenizer, max_seq_length, is_training, output_fn):
  """Loads a data file into tf record."""

  with tf.gfile.Open(file_name, 'r') as reader:
    for line in reader.readlines():
      cur_data = json.loads(line)

      tokens = []
      segment_ids = []
      labels = []
      label_ids = []

      tokens.append("[CLS]")
      segment_ids.append(0)

      for t in cur_data["chars"][0: max_seq_length - 1]:
        tokens.append(t)
        segment_ids.append(0)

      input_ids = text_tokenizer.convert_tokens_to_ids(tokens)
      input_mask = [1] * len(input_ids)

      if is_training:
        # no label for CLS
        for l in cur_data["labels"][0: max_seq_length]:
          labels.append(l)
        while len(labels) < max_seq_length:
          labels.append("O")
        label_ids = label_tokenizer.convert_tokens_to_ids(labels)

      # Zero-pad up to the sequence length.
      while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

      assert len(input_ids) == max_seq_length
      assert len(input_mask) == max_seq_length
      assert len(segment_ids) == max_seq_length

      feature = InputFeatures(tokens, input_ids, input_mask, segment_ids, labels, label_ids)

      # Run callback
      output_fn(feature)


def input_fn_builder(input_file, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
    "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
    "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
    "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
  }

  if is_training:
    name_to_features["label_ids"] = tf.FixedLenFeature([seq_length], tf.int64)

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      tf.logging.info("============_decode_record:%s", name)
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
      tf.contrib.data.map_and_batch(
        lambda record: _decode_record(record, name_to_features),
        batch_size=batch_size,
        drop_remainder=drop_remainder))

    return d

  return input_fn

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, sequence_lengths, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
    config=bert_config,
    is_training=is_training,
    input_ids=input_ids,
    input_mask=input_mask,
    token_type_ids=segment_ids,
    use_one_hot_embeddings=use_one_hot_embeddings)

  final_hidden = model.get_sequence_output()

  final_hidden_shape = modeling.get_shape_list(final_hidden, expected_rank=3)
  batch_size = final_hidden_shape[0]
  seq_length = final_hidden_shape[1]
  hidden_size = final_hidden_shape[2]

  output_weights = tf.get_variable(
    "cls/squad/output_weights", [num_labels, hidden_size],
    initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
    "cls/squad/output_bias", [num_labels], initializer=tf.zeros_initializer())

  final_hidden_matrix = tf.reshape(final_hidden,
                                   [batch_size * seq_length, hidden_size])
  logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
  logits = tf.nn.bias_add(logits, output_bias)

  logits = tf.reshape(logits, [-1, seq_length, num_labels])

  def __loss_layer(logits):
    with tf.variable_scope("crf_loss"):
      trans = tf.get_variable(
        "transitions",
        shape=[num_labels, num_labels],
        initializer=tf.zeros_initializer())

      log_likelihood, trans = tensorflow.contrib.crf.crf_log_likelihood(
        inputs=logits,
        tag_indices=labels,
        transition_params=trans,
        sequence_lengths=sequence_lengths)

      per_example_loss = -log_likelihood
      total_loss = tf.reduce_mean(per_example_loss)

      return (trans, per_example_loss, total_loss)

  trans, per_example_loss, total_loss = __loss_layer(logits)

  return (logits, trans, per_example_loss, total_loss)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    used = tf.sign(tf.abs(input_ids))
    length = tf.reduce_sum(used, reduction_indices=1)
    sequence_lengths = tf.cast(length, tf.int32)

    (logits, trans, per_example_loss, total_loss) = create_model(
      bert_config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      labels=label_ids,
      num_labels=num_labels,
      sequence_lengths=sequence_lengths,
      use_one_hot_embeddings=use_one_hot_embeddings)

    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
       ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    def decode(logits, lengths, trans):
      # inference final labels usa viterbi Algorithm
      paths = []
      for score, length in zip(logits, lengths):
        score = score[:length]
        path, _ = tensorflow.contrib.crf.viterbi_decode(score, trans)
        paths.append(path)
      return paths

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
        total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=total_loss,
        train_op=train_op,
        scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
      def metric_fn(per_example_loss, label_ids, logits, trans):
        predict_labels = decode(logits, sequence_lengths, trans)
        accuracy = tf.metrics.accuracy(label_ids, predict_labels)
        loss = tf.metrics.mean(per_example_loss)
        return {
          "eval_accuracy": accuracy,
          "eval_loss": loss,
        }

      eval_metrics = (metric_fn, [per_example_loss, label_ids, logits, trans])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode,
        loss=total_loss,
        eval_metrics=eval_metrics,
        scaffold_fn=scaffold_fn)
    else:
      predict_labels = decode(logits, sequence_lengths, trans)
      predictions = {
        "input_ids": input_ids,
        "predict_ids": predict_labels,
      }
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
        mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
      "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
      "Cannot use sequence length %d because the BERT model "
      "was only trained up to sequence length %d" %
      (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  text_tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  label_tokenizer = tokenization.FullTokenizer(
    vocab_file=FLAGS.label_vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
    cluster=tpu_cluster_resolver,
    master=FLAGS.master,
    model_dir=FLAGS.output_dir,
    save_checkpoints_steps=FLAGS.save_checkpoints_steps,
    tpu_config=tf.contrib.tpu.TPUConfig(
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_shards=FLAGS.num_tpu_cores,
      per_host_input_for_training=is_per_host))

  num_train_steps = None
  num_warmup_steps = None
  example_count = 0

  if FLAGS.do_train:
    example_count = parse_file_len(FLAGS.train_file)
    num_train_steps = int(example_count / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
    bert_config=bert_config,
    num_labels=len(label_tokenizer.vocab),
    init_checkpoint=FLAGS.init_checkpoint,
    learning_rate=FLAGS.learning_rate,
    num_train_steps=num_train_steps,
    num_warmup_steps=num_warmup_steps,
    use_tpu=FLAGS.use_tpu,
    use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    # We write to a temporary file to avoid storing very large constant tensors
    # in memory.
    train_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "train.tf_record"),
      is_training=True)
    read_ner_features(
      file_name=FLAGS.train_file,
      text_tokenizer=text_tokenizer,
      label_tokenizer=label_tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      is_training=True,
      output_fn=train_writer.process_feature)
    train_writer.close()

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num orig examples = %d", example_count)
    tf.logging.info("  Num split examples = %d", train_writer.num_features)
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)

    train_input_fn = input_fn_builder(
      input_file=train_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=True,
      drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_predict:
    example_count = parse_file_len(FLAGS.predict_file)
    eval_writer = FeatureWriter(
      filename=os.path.join(FLAGS.output_dir, "eval.tf_record"),
      is_training=False)
    eval_features = []

    def append_feature(feature):
      eval_features.append(feature)
      eval_writer.process_feature(feature)

    read_ner_features(
      file_name=FLAGS.predict_file,
      text_tokenizer=text_tokenizer,
      label_tokenizer=label_tokenizer,
      max_seq_length=FLAGS.max_seq_length,
      is_training=False,
      output_fn=append_feature)
    eval_writer.close()

    tf.logging.info("***** Running predictions *****")
    tf.logging.info("  Num orig examples = %d", example_count)
    tf.logging.info("  Num split examples = %d", len(eval_features))
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    all_results = []
    predict_input_fn = input_fn_builder(
      input_file=eval_writer.filename,
      seq_length=FLAGS.max_seq_length,
      is_training=False,
      drop_remainder=False)

    # If running eval on the TPU, you will need to specify the number of
    # steps.
    for result in estimator.predict(
            predict_input_fn, yield_single_examples=True):
      if len(all_results) % 1000 == 0:
        tf.logging.info("Processing example: %d" % (len(all_results)))
      input_ids = [float(x) for x in result["input_ids"].flat]
      predict_ids = [float(x) for x in result["predict_ids"].flat]
      all_results.append(RawResult(
        input_ids=input_ids,
        predict_ids=predict_ids))

    output_prediction_file = os.path.join(FLAGS.output_dir, "predictions.json")

    write_predictions(text_tokenizer.inv_vocab, all_results, output_prediction_file)


if __name__ == "__main__":
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("label_vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()
