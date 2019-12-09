import os
import argparse
import logging
import sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import rnn_cell_impl
import pandas as pd

from utils import createVocabulary, loadVocabulary, computeF1Score, DataProcessor, load_embedding, build_embedd_table
from unk_enhance import*

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--num_units", type=int, default=64, help="Network size. Actually, it's the dim of hidden reprensentation", dest='layer_size')
parser.add_argument("--model_type", type=str, default='full', help="""full(default) | intent_only
                                                                    full: full attention model
                                                                    intent_only: intent attention model""")
parser.add_argument("--priority_order", type=str, default='slot_first', help="""Type 'slot_first' or 'intent_first'
                                                                              to decide whose influence ought to calculate first use.""")
parser.add_argument("--use_crf", type=bool, default=False, help="""use crf for seq labeling""")
parser.add_argument("--use_embedding", type=str, default='1', help="""use pre-trained embedding""")
parser.add_argument("--cell", type=str, default='lstm', help="""rnn cell""")
parser.add_argument("--iteration_num", type=int, default=1, help="""the number of iteration times""")
parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
parser.add_argument("--batch_size_add", type=int, default=4, help="Batch size add.")
parser.add_argument("--max_epochs", type=int, default=100, help="Max epochs to train.")
parser.add_argument("--no_early_stop", action='store_false', dest='early_stop',
                    help="Disable early stop, which is based on sentence level accuracy.")
parser.add_argument("--patience", type=int, default=50, help="Patience to wait before stop.")
parser.add_argument("--learning_rate_decay", type=str, default='1', help="learning_rate_decay")
parser.add_argument("--learning_rate", type=float, default=0.001, help="The initial learning rate.")
parser.add_argument("--decay_steps", type=int, default=280 * 4, help="decay_steps.")
parser.add_argument("--decay_rate", type=float, default=0.9, help="decay_rate.")
parser.add_argument("--dataset", type=str, default='atis', help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")
parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
parser.add_argument("--valid_data_path", type=str, default='valid', help="Path to validation data files.")
parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")
parser.add_argument("--embedding_path", type=str, default='', help="embedding array's path.")
parser.add_argument("--embed_dim", type=int, default=300, help="Embedding dim.", dest='embed_dim')
parser.add_argument("--use_bert", type=bool, default=False, help="Use BERT embeddings.", dest='use_bert')
parser.add_argument("--use_unk", type=bool, default=False, help="to decide whether to use unk-enhanced data")
parser.add_argument("--unk_ratio", type=float, default='', help="unk_enhanced ratio")
parser.add_argument("--unk_threshold", type=int, default='', help="unk_enhanced threshold")
parser.add_argument("--unk_priority", type=str, default='', help="unk_enhanced priority. Only the following three options are available: full, entity, outside")
parser.add_argument("--intent_guide_slot", type=bool, default=False, help="decide whether to use predicted intent to help guide the prediction of slot prediction")
parser.add_argument("--bert_ip", type=str, default='', help="provide bert-server ip for bert client")


arg = parser.parse_args()

if arg.embedding_path == "" and arg.use_unk == False:
    arg.embed_dim = 64
elif arg.use_bert == True:
    arg.embed_dim = 1024
    

if arg.dataset == 'atis':
    arg.model_type = 'intent_only'
else:
    arg.model_type = 'full'

for k, v in sorted(vars(arg).items()):
    print(k, '=', v)
print()

if arg.model_type == 'full':
    remove_slot_attn = False
elif arg.model_type == 'intent_only':
    remove_slot_attn = True
else:
    print('unknown model type!')
    exit(1)

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

createVocabulary(os.path.join(full_train_path, arg.input_file), os.path.join(arg.vocab_path, 'in_vocab'))
createVocabulary(os.path.join(full_train_path, arg.slot_file), os.path.join(arg.vocab_path, 'slot_vocab'))
createVocabulary(os.path.join(full_train_path, arg.intent_file), os.path.join(arg.vocab_path, 'intent_vocab'),
                 no_pad=True)

in_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'in_vocab'))
slot_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'slot_vocab'))
intent_vocab = loadVocabulary(os.path.join(arg.vocab_path, 'intent_vocab'))


def create_full_vocabulary():
    # {'vocab': {'_PAD': 0, '_UNK': 1, 'to': 2, 'from': 3}, 'rev': ['_PAD', '_UNK', 'to', 'from']}
    word_alphabet = in_vocab["rev"]
    return np.array(word_alphabet)


def createModel(input_data, in_vocabulary_size, sequence_length, slots, slot_size, intent_size, layer_size=128,
                isTraining=True, embed_dim=64):
    cell_fw = tf.contrib.rnn.BasicLSTMCell(layer_size)
    cell_bw = tf.contrib.rnn.BasicLSTMCell(layer_size)

    if isTraining == True:
        cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, input_keep_prob=0.5,
                                                output_keep_prob=0.5)
        cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, input_keep_prob=0.5,
                                                output_keep_prob=0.5)

    if arg.use_bert:
        # we already have the embeddings in this case
        inputs = input_data
    else:
        if arg.embedding_path:
            embeddings_dict = load_embedding(arg.embedding_path)
            word_alphabet = create_full_vocabulary()
            embeddings_weight = build_embedd_table(word_alphabet, embeddings_dict, embedd_dim=embed_dim, caseless=True)
            embedding = tf.get_variable(name="embedding", shape=embeddings_weight.shape,
                                        initializer=tf.constant_initializer(embeddings_weight),
                                        trainable=True)
        else:
            embedding = tf.get_variable('embedding', [in_vocabulary_size, embed_dim])
        print("embedding shape", embedding.shape)
        inputs = tf.nn.embedding_lookup(embedding, input_data)

    state_outputs, final_state = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                                 sequence_length=sequence_length, dtype=tf.float32)
    final_state = tf.concat([final_state[0][0], final_state[0][1], final_state[1][0], final_state[1][1]], 1)
    state_outputs = tf.concat([state_outputs[0], state_outputs[1]], 2)
    state_shape = state_outputs.get_shape()

    with tf.variable_scope('attention'):
        slot_inputs = state_outputs
        if not remove_slot_attn:
            with tf.variable_scope('slot_attn'):
                attn_size = state_shape[2].value
                origin_shape = tf.shape(state_outputs)
                hidden = tf.expand_dims(state_outputs, 1)
                hidden_conv = tf.expand_dims(state_outputs, 2)
                k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
                hidden_features = tf.nn.conv2d(hidden_conv, k, [1, 1, 1, 1], "SAME")
                hidden_features = tf.reshape(hidden_features, origin_shape)
                hidden_features = tf.expand_dims(hidden_features, 1)
                v = tf.get_variable("AttnV", [attn_size])
                slot_inputs_shape = tf.shape(slot_inputs)
                slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])
                y = core_rnn_cell._linear(slot_inputs, attn_size, True)
                y = tf.reshape(y, slot_inputs_shape)
                y = tf.expand_dims(y, 2)
                s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [3])
                a = tf.nn.softmax(s)
                a = tf.expand_dims(a, -1)
                slot_d = tf.reduce_sum(a * hidden, [2])
                slot_reinforce_state = tf.expand_dims(slot_d, 2)
        else:
            attn_size = state_shape[2].value
            slot_d = slot_inputs
            slot_reinforce_state = tf.expand_dims(slot_inputs, 2)
            slot_inputs = tf.reshape(slot_inputs, [-1, attn_size])

        intent_input = final_state
        with tf.variable_scope('intent_attn'):
            attn_size = state_shape[2].value
            hidden = tf.expand_dims(state_outputs, 2)
            k = tf.get_variable("AttnW", [1, 1, attn_size, attn_size])
            hidden_features = tf.nn.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
            v = tf.get_variable("AttnV", [attn_size])

            y = core_rnn_cell._linear(intent_input, attn_size, True)
            y = tf.reshape(y, [-1, 1, 1, attn_size])
            s = tf.reduce_sum(v * tf.tanh(hidden_features + y), [2, 3])
            a = tf.nn.softmax(s)
            a = tf.expand_dims(a, -1)
            a = tf.expand_dims(a, -1)
            d = tf.reduce_sum(a * hidden, [1, 2])
            r_intent = d
            intent_context_states = d

        if arg.priority_order == 'intent_first':
            for n in range(arg.iteration_num):
                with tf.variable_scope('intent_subnet' + str(n - 1)):
                    attn_size = state_shape[2].value
                    hidden = tf.expand_dims(state_outputs, 2)
                    k1 = tf.get_variable("W1", [1, 1, attn_size, attn_size])
                    k2 = tf.get_variable('W2', [1, 1, attn_size, attn_size])
                    slot_reinforce_features = tf.nn.conv2d(slot_reinforce_state, k1, [1, 1, 1, 1],
                                                           "SAME")
                    hidden_features = tf.nn.conv2d(hidden, k2, [1, 1, 1, 1], "SAME")
                    v1 = tf.get_variable("AttnV", [attn_size])
                    bias = tf.get_variable("Bias", [attn_size])
                    s = tf.reduce_sum(v1 * tf.tanh(hidden_features + slot_reinforce_features + bias), [2, 3])
                    a = tf.nn.softmax(s)
                    a = tf.expand_dims(a, -1)
                    a = tf.expand_dims(a, -1)
                    r = tf.reduce_sum(a * slot_reinforce_state, [1, 2])

                    r_intent = r + intent_context_states

                    intent_output = tf.concat([r_intent, intent_input], 1)

                with tf.variable_scope('slot_subnet' + str(n - 1)):
                    intent_gate = core_rnn_cell._linear(r_intent, attn_size, True)
                    intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[
                        1].value])
                    v1 = tf.get_variable("gateV", [attn_size])
                    relation_factor = v1 * tf.tanh(slot_d + intent_gate)
                    relation_factor = tf.reduce_sum(relation_factor, [2])
                    relation_factor = tf.expand_dims(relation_factor, -1)
                    slot_reinforce_state1 = slot_d * relation_factor
                    slot_reinforce_state = tf.expand_dims(slot_reinforce_state1, 2)
                    slot_reinforce_vector = tf.reshape(slot_reinforce_state1, [-1, attn_size])
                    slot_output = tf.concat([slot_reinforce_vector, slot_inputs], 1)

        else:
            for n in range(arg.iteration_num):
                with tf.variable_scope('slot_subnet' + str(n - 1)):
                    intent_gate = core_rnn_cell._linear(r_intent, attn_size, True)
                    intent_gate = tf.reshape(intent_gate, [-1, 1, intent_gate.get_shape()[
                        1].value])
                    v1 = tf.get_variable("gateV", [attn_size])
                    relation_factor = v1 * tf.tanh(slot_d + intent_gate)
                    relation_factor = tf.reduce_sum(relation_factor, [2])
                    relation_factor = tf.expand_dims(relation_factor, -1)
                    slot_reinforce_state = slot_d * relation_factor
                    slot_reinforce_vector = tf.reshape(slot_reinforce_state, [-1, attn_size])
                    slot_output = tf.concat([slot_reinforce_vector, slot_inputs], 1)

                with tf.variable_scope('intent_subnet' + str(n - 1)):
                    attn_size = state_shape[2].value
                    hidden = tf.expand_dims(state_outputs, 2)
                    slot_reinforce_output = tf.expand_dims(slot_reinforce_state, 2)
                    k1 = tf.get_variable("W1", [1, 1, attn_size, attn_size])
                    k2 = tf.get_variable('W2', [1, 1, attn_size, attn_size])
                    slot_features = tf.nn.conv2d(slot_reinforce_output, k1, [1, 1, 1, 1], "SAME")
                    hidden_features = tf.nn.conv2d(hidden, k2, [1, 1, 1, 1], "SAME")
                    v1 = tf.get_variable("AttnV", [attn_size])
                    bias = tf.get_variable("Bias", [attn_size])
                    s = tf.reduce_sum(v1 * tf.tanh(hidden_features + slot_features + bias), [2, 3])
                    a = tf.nn.softmax(s)
                    a = tf.expand_dims(a, -1)
                    a = tf.expand_dims(a, -1)
                    r = tf.reduce_sum(a * slot_reinforce_output, [1, 2])

                    r_intent = r + intent_context_states

                    intent_output = tf.concat([r_intent, intent_input], 1)
        # if arg.intent_guide_slot == True:
        #     print(slot_output)
        #     print(intent_output)
        #     slot_output = tf.concat([slot_output, intent_output], 1)

    with tf.variable_scope('intent_proj'):
        intent = core_rnn_cell._linear(intent_output, intent_size, True)
    with tf.variable_scope('slot_proj'):
        slot = core_rnn_cell._linear(slot_output, slot_size, True)
        if arg.use_crf:
            nstep = tf.shape(state_outputs)[1]
            slot = tf.reshape(slot, [-1, nstep, slot_size])
    outputs = [slot, intent]
    return outputs


input_data = tf.placeholder(tf.int32, [None, None], name='inputs')
input_sequence_embeddings = tf.placeholder(tf.float32, [None, None, arg.embed_dim], name='input_sequence_embeddings')
sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
global_step = tf.Variable(0, trainable=False, name='global_step')
slots = tf.placeholder(tf.int32, [None, None], name='slots')
slot_weights = tf.placeholder(tf.float32, [None, None], name='slot_weights')
intent = tf.placeholder(tf.int32, [None], name='intent')

with tf.variable_scope('model'):
    input_raw = input_sequence_embeddings if arg.use_bert else input_data
    training_outputs = createModel(input_raw, len(in_vocab['vocab']), sequence_length, slots, len(slot_vocab['vocab']),
                                   len(intent_vocab['vocab']), layer_size=arg.layer_size, embed_dim=arg.embed_dim)


slots_shape = tf.shape(slots)
slots_reshape = tf.reshape(slots, [-1])

slot_outputs = training_outputs[0]
with tf.variable_scope('slot_loss'):
    if arg.use_crf:
        log_likelihood, trans_params = tf.contrib.crf.crf_log_likelihood(slot_outputs, slots, sequence_length)
        slot_loss = tf.reduce_mean(-log_likelihood)
    else:
        crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slots_reshape, logits=slot_outputs)
        crossent = tf.reshape(crossent, slots_shape)
        slot_loss = tf.reduce_sum(crossent * slot_weights, 1)
        total_size = tf.reduce_sum(slot_weights, 1)
        total_size += 1e-12
        slot_loss = slot_loss / total_size

intent_output = training_outputs[1]
with tf.variable_scope('intent_loss'):
    crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_output)
    intent_loss = tf.reduce_sum(crossent) / tf.cast(arg.batch_size, tf.float32)
params = tf.trainable_variables()
learning_rate = tf.train.exponential_decay(arg.learning_rate, global_step, arg.decay_steps, arg.decay_rate,
                                           staircase=False)
if arg.learning_rate_decay:
    opt = tf.train.AdamOptimizer(learning_rate)
else:
    opt = tf.train.AdamOptimizer(arg.learning_rate)
intent_params = []
slot_params = []
for p in params:
    if not 'slot_' in p.name:
        intent_params.append(p)
    if 'slot_' in p.name or 'bidirectional_rnn' in p.name or 'embedding' in p.name:
        slot_params.append(p)

gradients_slot = tf.gradients(slot_loss, slot_params)
gradients_intent = tf.gradients(intent_loss, intent_params)

clipped_gradients_slot, norm_slot = tf.clip_by_global_norm(gradients_slot, 5.0)
clipped_gradients_intent, norm_intent = tf.clip_by_global_norm(gradients_intent, 5.0)

gradient_norm_slot = norm_slot
gradient_norm_intent = norm_intent
update_slot = opt.apply_gradients(zip(clipped_gradients_slot, slot_params))
update_intent = opt.apply_gradients(zip(clipped_gradients_intent, intent_params), global_step=global_step)

training_outputs = [global_step, slot_loss, update_intent, update_slot, gradient_norm_intent, gradient_norm_slot]
inputs = [input_data, sequence_length, slots, slot_weights, intent]


with tf.variable_scope('model', reuse=True):
    input_raw = input_sequence_embeddings if arg.use_bert else input_data
    inference_outputs = createModel(input_raw, len(in_vocab['vocab']), sequence_length, slots,
                                    len(slot_vocab['vocab']), len(intent_vocab['vocab']),
                                    layer_size=arg.layer_size, isTraining=False, embed_dim=arg.embed_dim)

if arg.use_crf:
    inference_slot_output, pred_scores = tf.contrib.crf.crf_decode(inference_outputs[0], trans_params, sequence_length)
else:
    inference_slot_output = tf.nn.softmax(inference_outputs[0], name='slot_output')

inference_intent_output = tf.nn.softmax(inference_outputs[1], name='intent_output')

inference_outputs = [inference_intent_output, inference_slot_output]
inference_inputs = [input_data, sequence_length]

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

saver = tf.train.Saver()
gpu_options = tf.GPUOptions(allow_growth=True)


def save_current_results(epoch, records):
    logging.info("Saving results of Epoch {}".format(str(epoch)))
    columns = ["split", "step", "epochs", "eval_loss", "slot_f1", "intent_accuracy", "semantic_accuracy"]
    df = pd.DataFrame(records, columns=columns)
    df.to_csv("./training_output.csv", index=False)
    logging.info("Results saved!")


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


with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    sess.run(tf.global_variables_initializer())
    logging.info('Training Start')

    epochs = 0
    loss = 0.0
    data_processor = None
    line = 0
    num_loss = 0
    step = 0
    no_improve = 0

    valid_slot = 0
    test_slot = 0
    valid_intent = 0
    test_intent = 0
    valid_err = 0
    test_err = 0
    best_epoch_num = 0
    eval_loss = 0.0

    result_records = []

    if arg.use_bert:
        from bert_serving.client import BertClient
        bc = BertClient(ip=arg.bert_ip)

    while True:
        if data_processor == None:
            
            # For unk purpose
            if arg.use_unk == True:
                unker = UNKer(os.path.join(full_train_path, arg.input_file), os.path.join(full_train_path, arg.input_file+".unk."+arg.unk_priority), os.path.join(full_train_path, arg.slot_file), 
                                            ratio=arg.unk_ratio, threshold=arg.unk_threshold, priority=arg.unk_priority)
                data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file+".unk."+arg.unk_priority),
                    os.path.join(full_train_path, arg.slot_file),
                    os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab,
                                intent_vocab, use_bert=arg.use_bert)
            else:
                data_processor = DataProcessor(os.path.join(full_train_path, arg.input_file),
                    os.path.join(full_train_path, arg.slot_file),
                    os.path.join(full_train_path, arg.intent_file), in_vocab, slot_vocab,
                                intent_vocab, use_bert=arg.use_bert)


        in_data, slot_data, slot_weight, length, intents, input_seq, _, _ = data_processor.get_batch(arg.batch_size)
        input_seq_embeddings = np.empty(shape=[0, 0, arg.embed_dim])

        if arg.use_bert:
            input_seq_embeddings = get_bert_embeddings(input_seq)

        feed_dict = {input_data.name: in_data, slots.name: slot_data, slot_weights.name: slot_weight,
                     sequence_length.name: length, intent.name: intents,
                     input_sequence_embeddings.name: input_seq_embeddings}
        ret = sess.run(training_outputs, feed_dict)

        if len(in_data) != 0:
            loss += np.mean(ret[1])
            line += arg.batch_size
            step = ret[0]
            num_loss += 1

        if data_processor.end == 1:
            arg.batch_size += arg.batch_size_add
            line = 0
            data_processor.close()
            data_processor = None
            epochs += 1
            logging.info('Step: ' + str(step))
            logging.info('Epochs: ' + str(epochs))
            logging.info('Loss: ' + str(loss / num_loss))
            eval_loss = loss / num_loss
            num_loss = 0
            loss = 0.0

            save_path = os.path.join(arg.model_path, '_step_' + str(step) + '_epochs_' + str(epochs) + '.ckpt')
            saver.save(sess, save_path)

            def valid(in_path, slot_path, intent_path):
                data_processor_valid = DataProcessor(in_path, slot_path, intent_path, in_vocab, slot_vocab,
                                                     intent_vocab, use_bert=arg.use_bert)

                pred_intents = []
                correct_intents = []
                slot_outputs = []
                correct_slots = []
                input_words = []

                gate_seq = []
                while True:
                    in_data, slot_data, slot_weight, length, intents, in_seq, slot_seq, intent_seq = data_processor_valid.get_batch(
                        arg.batch_size)
                    if len(in_data) <= 0:
                        break

                    input_seq_embeddings = np.empty(shape=[0, 0, arg.embed_dim])
                    if arg.use_bert:
                        input_seq_embeddings = get_bert_embeddings(in_seq)

                    feed_dict = {input_data.name: in_data, sequence_length.name: length,
                                 input_sequence_embeddings.name: input_seq_embeddings}
                    ret = sess.run(inference_outputs, feed_dict)
    
                    for i in ret[0]:
                        pred_intents.append(np.argmax(i))
                    for i in intents:
                        correct_intents.append(i)

                    pred_slots = ret[1].reshape((slot_data.shape[0], slot_data.shape[1], -1))
                    for p, t, i, l in zip(pred_slots, slot_data, in_data, length):
                        if arg.use_crf:
                            p = p.reshape([-1])
                        else:
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
                accuracy = (pred_intents == correct_intents)
                semantic_acc = accuracy
                accuracy = accuracy.astype(float)
                accuracy = np.mean(accuracy) * 100.0

                index = 0
                for t, p in zip(correct_slots, slot_outputs):
                    # Process Semantic Error
                    if len(t) != len(p):
                        raise ValueError('Error!!')

                    for j in range(len(t)):
                        if p[j] != t[j]:
                            semantic_acc[index] = False
                            break
                    index += 1
                semantic_acc = semantic_acc.astype(float)
                semantic_acc = np.mean(semantic_acc) * 100.0

                f1, precision, recall = computeF1Score(correct_slots, slot_outputs)
                logging.info('slot f1: ' + str(f1))
                logging.info('intent accuracy: ' + str(accuracy))
                logging.info('semantic Acc(intent, slots are all correct): ' + str(semantic_acc))

                data_processor_valid.close()
                return f1, accuracy, semantic_acc, pred_intents, correct_intents, slot_outputs, correct_slots, input_words, gate_seq

            logging.info('Valid:')
            epoch_valid_slot, epoch_valid_intent, epoch_valid_err, valid_pred_intent, valid_correct_intent, valid_pred_slot, valid_correct_slot, valid_words, valid_gate = valid(
                os.path.join(full_valid_path, arg.input_file), os.path.join(full_valid_path, arg.slot_file),
                os.path.join(full_valid_path, arg.intent_file))

            result_records.append(["valid", step, epochs, eval_loss, epoch_valid_slot, epoch_valid_intent, epoch_valid_err])

            logging.info('Test:')
            epoch_test_slot, epoch_test_intent, epoch_test_err, test_pred_intent, test_correct_intent, test_pred_slot, test_correct_slot, test_words, test_gate = valid(
                os.path.join(full_test_path, arg.input_file), os.path.join(full_test_path, arg.slot_file),
                os.path.join(full_test_path, arg.intent_file))

            result_records.append(["test", step, epochs, -1, epoch_test_slot, epoch_test_intent, epoch_test_err])

            save_current_results(epochs, result_records)

            if epoch_test_err <= test_err:
                no_improve += 1
            else:
                best_epoch_num = epochs
                test_err = epoch_test_err

                no_improve = 0

            if test_err > 0:
                logging.info('best epoch_num :  {}'.format(best_epoch_num))
                logging.info('best score : {}'.format(test_err))

            if epochs == arg.max_epochs:
                break

            if arg.early_stop == True:
                if no_improve > arg.patience:
                    break
