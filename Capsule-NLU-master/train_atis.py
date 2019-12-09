# -*- coding: utf-8 -*-
import argparse
import logging
import os

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf

#from create_model import build_model
from utils import margin_loss
from capsule_masked import Capsule

from utils import createVocabulary, loadVocabulary, computeF1Score, DataProcessor, load_embedding, build_embedd_table
#Unk
from unk_enhance import*

# Processing Units logs
log_device_placement = False
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser(allow_abbrev=False)
# Network
parser.add_argument("--num_units", type=int, default=512, help="Network size.", dest='layer_size')
parser.add_argument("--embed_dim", type=int, default=1024, help="Embedding dim.", dest='embed_dim')
parser.add_argument("--intent_dim", type=int, default=128, help="Intent dim.", dest='intent_dim')
parser.add_argument("--model_type", type=str, default='full', help="""full(default) | without_rerouting.
                                                                    full: full model with re-routing
                                                                    without_rerouting: model without re-routing""")
parser.add_argument("--num_rnn", type=int, default=1, help="Num of layers for stacked RNNs.")
parser.add_argument("--iter_slot", type=int, default=2, help="Num of iteration for slots.")
parser.add_argument("--iter_intent", type=int, default=2, help="Num of iteration for intents.")

# Training Environment
parser.add_argument("--optimizer", type=str, default='rmsprop', help="Optimizer.")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Batch size.")
parser.add_argument("--margin", type=float, default=0.4, help="Margin in the max-margin loss.")
parser.add_argument("--downweight", type=float, default=0.5, help="Downweight for the max-margin loss.")
parser.add_argument("--max_epochs", type=int, default=60, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false', dest='early_stop',
                    help="Disable early stop, which is based on sentence level accuracy.")
parser.add_argument("--patience", type=int, default=40, help="Patience to wait before stop.")
parser.add_argument("--run_name", type=str, default='capsule_nlu', help="Run name.")


#Bert
parser.add_argument("--use_bert", type=bool, default=False, help="Use BERT embeddings.", dest='use_bert')
parser.add_argument("--bert_ip", type=str, default='', help="provide bert-server ip for bert client")

#Embedding
parser.add_argument("--use_embedding", type=str, default='1', help="""use pre-trained embedding""")
parser.add_argument("--embedding_path", type=str, default='../../nqg/glove.840B.300d.txt')

# Model and Data
parser.add_argument("--dataset", type=str, default='atis', help="""Type 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")

##
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

#Unk
parser.add_argument("--use_unk", type=bool, default=False, help="to decide whether to use unk-enhanced data")
parser.add_argument("--unk_ratio", type=float, default=0.25, help="unk_enhanced ratio")
parser.add_argument("--unk_threshold", type=int, default=20, help="unk_enhanced threshold")
parser.add_argument("--unk_priority", type=str, default='entity', help="unk_enhanced priority. Only the following three options are available: full, entity, outside")


arg = parser.parse_args()
logs_path = './log/' + arg.run_name

# Print arguments
for k, v in sorted(vars(arg).items()):
    print(k, '=', v)
print()

# Optimzers
if arg.optimizer == 'adam':
    opt = tf.train.AdamOptimizer(learning_rate=arg.learning_rate)
elif arg.optimizer == 'rmsprop':
    opt = tf.train.RMSPropOptimizer(learning_rate=arg.learning_rate)
elif arg.optimizer == 'adadelta':
    opt = tf.train.AdadeltaOptimizer(learning_rate=arg.learning_rate)
elif arg.optimizer == 'adagrad':
    opt = tf.train.AdagradOptimizer(learning_rate=arg.learning_rate)
else:
    print('unknown optimizer!')
    exit(1)

# Ablation
if arg.model_type == 'full':
    re_routing = True
elif arg.model_type == 'without_rerouting':
    re_routing = False
else:
    print('unknown model type!')
    exit(1)

# Full path to data will be: ./data/ + dataset + train/test/valid
if arg.dataset == None:
    print('name of dataset can not be None')
    exit(1)
elif arg.dataset == 'snips':
    print('use snips dataset')
elif arg.dataset == 'atis':
    print('use atis dataset')
else:
    print('use own dataset: ', arg.dataset)
full_train_path = os.path.join('./data', arg.dataset, arg.train_data_path)
full_test_path = os.path.join('./data', arg.dataset, arg.test_data_path)
full_valid_path = os.path.join('./data', arg.dataset, arg.valid_data_path)

# Create vocabulary and save vocab files in ./vocab
createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'),
                 pad=False, unk=False)

# Load vocab
in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))
intent_dim = arg.intent_dim


# Create training model
def build_model(input_data, input_size, sequence_length, slot_size, intent_size, intent_dim, layer_size, embed_dim,
                num_rnn=1, isTraining=True, iter_slot=2, iter_intent=2, re_routing=True):
    cell_fw_list = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(layer_size) for _ in range(num_rnn)])
    cell_bw_list = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.BasicLSTMCell(layer_size) for _ in range(num_rnn)])

    if isTraining == True:
        cell_fw_list = tf.contrib.rnn.DropoutWrapper(cell_fw_list, input_keep_prob=0.8,
                                                     output_keep_prob=0.8)
        cell_bw_list = tf.contrib.rnn.DropoutWrapper(cell_bw_list, input_keep_prob=0.8,
                                                     output_keep_prob=0.8)

    #Bert
    if arg.use_bert:
        # we already have the embeddings in this case
        inputs = input_data
    else:
        embedding = tf.get_variable('embedding', [input_size, embed_dim],
                                initializer=tf.contrib.layers.xavier_initializer())
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    with tf.variable_scope('slot_capsule'):
        H, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
            [cell_fw_list],
            [cell_bw_list],
            inputs=inputs,
            sequence_length=sequence_length,
            dtype=tf.float32)
        sc = Capsule(slot_size, layer_size, reuse=tf.AUTO_REUSE, iter_num=iter_slot, wrr_dim=(layer_size, intent_dim))
        slot_capsule, routing_weight, routing_logits = sc(H, sequence_length, re_routing=False)
    with tf.variable_scope('slot_proj'):
        slot_p = tf.reshape(routing_logits, [-1, slot_size])
    with tf.variable_scope('intent_capsule'):
        intent_capsule, intent_routing_weight, _ = Capsule(intent_size, intent_dim, reuse=tf.AUTO_REUSE,
                                                           iter_num=iter_intent)(slot_capsule, slot_size)
    with tf.variable_scope('intent_proj'):
        intent = intent_capsule
    outputs = [slot_p, intent, routing_weight, intent_routing_weight]
    if re_routing:
        pred_intent_index_onehot = tf.one_hot(tf.argmax(tf.norm(intent_capsule, axis=-1), axis=-1), intent_size)
        pred_intent_index_onehot = tf.tile(tf.expand_dims(pred_intent_index_onehot, 2),
                                           [1, 1, tf.shape(intent_capsule)[2]])
        intent_capsule_max = tf.reduce_sum(tf.multiply(intent_capsule, tf.cast(pred_intent_index_onehot, tf.float32)),
                                           axis=1,
                                           keepdims=False)
        caps_ihat = tf.expand_dims(tf.expand_dims(intent_capsule_max, 1), 3)
        with tf.variable_scope('slot_capsule', reuse=True):
            slot_capsule_new, routing_weight_new, routing_logits_new = sc(H, sequence_length, caps_ihat=caps_ihat,
                                                                          re_routing=True)
        with tf.variable_scope('slot_proj', reuse=True):
            slot_p_new = tf.reshape(routing_logits_new, [-1, slot_size])
        outputs = [slot_p_new, intent, routing_weight_new, intent_routing_weight]
    return outputs



input_data = tf.placeholder(tf.int32, [None, None], name='inputs')  # word ids
input_sequence_embeddings = tf.placeholder(tf.float32, [None, None, arg.embed_dim], name='input_sequence_embeddings')
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")  # sequence length
global_step = tf.Variable(0, trainable=False, name='global_step')
slots = tf.placeholder(tf.int32, [None, None], name='slots')  # slot ids
slot_weights = tf.placeholder(tf.float32, [None, None], name='slot_weights')  # sequence mask
intent = tf.placeholder(tf.int32, [None], name='intent')  # intent label

with tf.variable_scope('model'):
    input_raw = input_sequence_embeddings if arg.use_bert else input_data
    training_outputs = build_model(input_raw, len(in_vocab['vocab']), sequence_length, len(slot_vocab['vocab']) - 2,
                                   len(intent_vocab['vocab']), intent_dim,
                                   layer_size=arg.layer_size, embed_dim=arg.embed_dim, num_rnn=arg.num_rnn,
                                   isTraining=True, iter_slot=arg.iter_slot, iter_intent=arg.iter_intent,
                                   re_routing=re_routing)

slots_shape = tf.shape(slots)
slots_reshape = tf.reshape(slots, [-1])
slot_outputs = training_outputs[0]
intent_outputs = training_outputs[1]
slot_routing_weight = training_outputs[2]
intent_routing_weight = training_outputs[3]
intent_outputs_norm = tf.norm(intent_outputs, axis=-1)

# Define slot loss
with tf.variable_scope('slot_loss'):
    slots_reshape_onehot = tf.one_hot(slots_reshape, len(slot_vocab['vocab']) - 2)  # [16*18, 74]
    crossent = tf.nn.softmax_cross_entropy_with_logits_v2(labels=slots_reshape_onehot, logits=slot_outputs)
    crossent = tf.reshape(crossent, slots_shape)
    slot_loss = tf.reduce_sum(crossent * slot_weights, 1)
    total_size = tf.reduce_sum(slot_weights, 1)
    total_size += 1e-12
    slot_loss = slot_loss / total_size

# Define intent loss
with tf.variable_scope('intent_loss'):
    intent_onehot = tf.one_hot(intent, len(intent_vocab['vocab']))
    marginloss = margin_loss(labels=intent_onehot, raw_logits=intent_outputs_norm, margin=arg.margin,
                             downweight=arg.downweight)
    intent_loss = tf.reduce_mean(marginloss, axis=-1)

# Specify the learning environment
params = tf.trainable_variables()
slot_params = []
for p in params:
    if 'slot' in p.name or 'embedding' in p.name:
        slot_params.append(p)
intent_params = []
for p in params:
    if 'intent' in p.name:
        intent_params.append(p)

gradients_slot = tf.gradients(slot_loss, slot_params)
gradients_intent = tf.gradients(intent_loss, intent_params)

clipped_gradients_slot, norm_slot = tf.clip_by_global_norm(gradients_slot, 5.0)
clipped_gradients_intent, norm_intent = tf.clip_by_global_norm(gradients_intent, 5.0)

gradient_norm_slot = norm_slot
gradient_norm_intent = norm_intent

update_slot = opt.apply_gradients(zip(clipped_gradients_slot, slot_params))
update_intent = opt.apply_gradients(zip(clipped_gradients_intent, intent_params), global_step=global_step)

training_outputs = [global_step, slot_loss, intent_loss, slot_routing_weight, intent_routing_weight, update_slot,
                    update_intent, gradient_norm_slot, gradient_norm_intent]
inputs = [input_data, sequence_length, slots, slot_weights, intent]

# Create Inference Model
with tf.variable_scope('model', reuse=True):
    input_raw = input_sequence_embeddings if arg.use_bert else input_data
    inference_outputs = build_model(input_raw, len(in_vocab['vocab']), sequence_length, len(slot_vocab['vocab']) - 2,
                                    len(intent_vocab['vocab']), intent_dim,
                                    layer_size=arg.layer_size, embed_dim=arg.embed_dim, num_rnn=arg.num_rnn,
                                    isTraining=False, iter_slot=arg.iter_slot, iter_intent=arg.iter_intent,
                                    re_routing=re_routing)

inference_intent_outputs_norm = tf.norm(inference_outputs[1], axis=-1)
inference_outputs = [inference_outputs[0], inference_outputs[1], inference_intent_outputs_norm, inference_outputs[2],
                     inference_outputs[3]]
inference_inputs = [input_data, sequence_length]

saver = tf.train.Saver()

def get_bert_embeddings(in_seq):
    input_seq_embeddings = bc.encode([s.split() for s in in_seq], is_tokenized=True).copy()
    dims = input_seq_embeddings.shape

    if dims[2] > arg.embed_dim:
        # if bert-service concatenated multiple layers, we reduce them to arg.embed_dim by summing them up.
        tmp_seq_embeddings = np.empty(shape=(dims[0], dims[1], arg.embed_dim))
        for i in range(dims[0]):
            for j in range(dims[1]):
                tmp_seq_embeddings[i][j] = np.sum(input_seq_embeddings[i][j].reshape(-1, arg.embed_dim), axis=0)
        input_seq_embeddings = tmp_seq_embeddings
    return input_seq_embeddings

# Start Training
with tf.Session(config=tf.ConfigProto(allow_soft_placement=False, log_device_placement=log_device_placement)) as sess:
    sess.run(tf.global_variables_initializer())
    logging.info('Training Start')
    epochs = 0
    eval_slot_loss = 0.0
    eval_intent_loss = 0.0
    eval_slot_p = 0.0
    data_processor = None
    line = 0
    num_loss = 0
    step = 0
    no_improve = 0

    # variables to store highest values among epochs, only use 'valid_err' for now
    valid_slot = 0
    test_slot = 0
    valid_intent = 0
    test_intent = 0
    valid_err = 0
    test_err = 0

    # Load from saved checkpoints
    # saver.restore(sess, './model/' + arg.run_name + ".ckpt")
    # logging.info("Model restored.")

    if arg.use_bert:
        from bert_serving.client import BertClient
        bc = BertClient()

    while True:
        if data_processor == None:
          
           #Unk
           # For unk purpose
            if arg.use_unk == True:
                unker = UNKer(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.input_file+".unk."+arg.unk_priority), os.path.join(full_train_path, arg.slot_file), 
                                            ratio=arg.unk_ratio, threshold=arg.unk_threshold, priority=arg.unk_priority)
                data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file+".unk."+arg.unk_priority),
                    os.path.join(full_train_path, arg.slot_file),
                    os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab,
                                intent_vocab, shuffle=True, use_bert=arg.use_bert)
            else:
                data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file),
                    os.path.join(full_train_path, arg.slot_file),
                    os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab,
                                intent_vocab, shuffle=True, use_bert=arg.use_bert)

        in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor.get_batch(arg.batch_size)
        input_seq_embeddings = np.empty(shape=[0, 0, arg.embed_dim])

        if arg.use_bert:
            input_seq_embeddings = get_bert_embeddings(input_seq)

        feed_dict = {input_data.name: in_data, slots.name: slot_data, slot_weights.name: slot_weight,
                     sequence_length.name: length, intent.name: intents,
                     input_sequence_embeddings.name: input_seq_embeddings}

        if len(in_data) != 0:
            ret = sess.run(training_outputs, feed_dict)
            eval_slot_loss += np.mean(ret[1])
            eval_intent_loss += np.mean(ret[2])
            line += len(in_data)
            step = ret[0]
            num_loss += 1

        if data_processor.end == 1:
            line = 0
            data_processor = None
            epochs += 1
            logging.info('Step: ' + str(step))
            logging.info('Epochs: ' + str(epochs))
            logging.info('Slot Loss: ' + str(eval_slot_loss / num_loss))
            logging.info('Intent Loss: ' + str(eval_intent_loss / num_loss))
            num_loss = 0
            eval_slot_loss = 0.0
            eval_slot_p = 0.0
            eval_intent_loss = 0.0
            save_path = os.path.join(arg.model_path, '_step_' + str(step) + '_epochs_' + str(epochs) + '.ckpt')


            def valid(in_path, slot_path, intent_path):
                data_processor_valid = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab,
                                                     intent_vocab, use_bert=arg.use_bert)

                pred_intents = []
                correct_intents = []
                slot_outputs = []
                correct_slots = []
                input_words = []

                while True:
                    in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(arg.batch_size)

                    input_seq_embeddings = np.empty(shape=[0, 0, arg.embed_dim])
                    if arg.use_bert:
                        input_seq_embeddings = get_bert_embeddings(in_seq)

                    feed_dict = {input_data.name: in_data, sequence_length.name: length,
                                 input_sequence_embeddings.name: input_seq_embeddings}

                    if len(in_data) != 0:
                        ret = sess.run(inference_outputs, feed_dict)
                        for i in ret[2]:
                            pred_intents.append(np.argmax(i))
                        for i in intents:
                            correct_intents.append(i)

                        pred_slots = ret[3][-1, :, :, :].reshape((slot_data.shape[0], slot_data.shape[1], -1))
                        for p, t, i, l, s in zip(pred_slots, slot_data, in_data, length, slot_seq):
                            p = np.argmax(p, 1)
                            tmp_pred = []
                            tmp_correct = []
                            tmp_input = []
                            for j in range(l):
                                tmp_pred.append(slot_vocab['rev'][p[j]])
                                tmp_correct.append(slot_vocab['rev'][t[j]])
                                tmp_input.append(in_vocab['rev'][i[j]])

                            slot_outputs.append(tmp_pred)
                            correct_slots.append(tmp_correct)
                            input_words.append(tmp_input)
                    if data_processor_valid.end == 1:
                        break
                pred_intents = np.array(pred_intents)
                correct_intents = np.array(correct_intents)
                from sklearn.metrics import classification_report
                logging.info(classification_report(y_true=correct_intents, y_pred=pred_intents, digits=4))
                accuracy = (pred_intents == correct_intents)
                semantic_error = accuracy
                accuracy = accuracy.astype(float)
                accuracy = np.mean(accuracy) * 100.0

                index = 0
                for t, p in zip(correct_slots, slot_outputs):
                    # Process Semantic Error
                    if len(t) != len(p):
                        raise ValueError('Error!!')

                    for j in range(len(t)):
                        if p[j] != t[j]:
                            semantic_error[index] = False
                            break
                    index += 1
                semantic_error = semantic_error.astype(float)
                semantic_error = np.mean(semantic_error) * 100.0

                f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
                logging.info('slot f1: ' + str(f1))
                logging.info('intent accuracy: ' + str(accuracy))
                logging.info('semantic error(intent, slots are all correct): ' + str(semantic_error))

                return f1, accuracy, semantic_error, pred_intents, correct_intents, slot_outputs, correct_slots, input_words


            logging.info('Valid:')
            epoch_valid_slot, epoch_valid_intent, epoch_valid_err, valid_pred_intent, valid_correct_intent, valid_pred_slot, valid_correct_slot, valid_words = valid(
                os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.slot_file),
                os.path.join(full_valid_path, arg.intent_file))

            logging.info('Test:')
            epoch_test_slot, epoch_test_intent, epoch_test_err, test_pred_intent, test_correct_intent, test_pred_slot, test_correct_slot, test_words = valid(
                os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.slot_file),
                os.path.join(full_test_path, arg.intent_file))

            if epoch_valid_err <= valid_err:
                no_improve += 1
            else:
                valid_err = epoch_valid_err
                no_improve = 0
            if epochs == arg.max_epochs:
                break
            if arg.early_stop:
                if no_improve > arg.patience:
                    break

            save_path = saver.save(sess, './model/' + arg.run_name + "_" + str(epochs) + ".ckpt")
            # logging.info("Model saved in path: " + str(save_path))