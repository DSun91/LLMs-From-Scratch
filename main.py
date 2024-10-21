import numpy as np
import jax.numpy as jnp
import re
import cupy as cp
import pickle
import time
import numpy as np
import jax.numpy as jnp
import pandas as pd
import numpy as np
import jax
import spacy
from tqdm import tqdm
from pathlib import Path
import json

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.3g" % x))


def log_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the wrapped function
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time
        # print(f"Function '{func.__name__}' executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


def get_positional_encoding(seq_len, d_model):
    """
    Returns a non-learnable (sinusoidal) positional encoding.


    seq_len: Length of the input sequence.
    d_model: Dimension of the embeddings.
    """
    pos = np.arange(seq_len)[:, np.newaxis]  # Shape: [seq_len, 1]
    i = np.arange(d_model)[np.newaxis, :]  # Shape: [1, d_model]

    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

    # Apply sine to even indices, cosine to odd indices
    pos_encoding = np.zeros((seq_len, d_model))
    pos_encoding[:, 0::2] = np.sin(pos * angle_rates[:, 0::2])  # sine on even indices
    pos_encoding[:, 1::2] = np.cos(pos * angle_rates[:, 1::2])  # cosine on odd indices

    return pos_encoding


def softmax(x, axis=-1):
    # Subtract the max value for numerical stability
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x, epsilon=1e-6):
    # Calculate the mean and variance
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)

    # Normalize the output
    x_norm = (x - mean) / jnp.sqrt(var + epsilon)
    # print(x)
    # print(mean)
    # print("mean",mean.shape)
    # print("x_norm.shape",x_norm.shape)
    return x_norm, mean, var, x.shape[-1]


def relu(x):
    return np.maximum(0, x)


# @log_time
def pad_sequence(seq, max_len, pad_value=0):
    """Pad a sequence with a given value up to max_len."""
    current_len = seq.shape[0]
    pad_width = max_len - current_len
    if pad_width > 0:
        # Pad sequence with zeros (or any pad_value you provide)
        seq = jnp.pad(seq, ((0, pad_width), (0, 0)), mode='constant', constant_values=pad_value)
    return seq


@log_time
def create_timestaped_input(input_d, words_per_phrase):
    input_translation = []
    for j in range(input_d.shape[0]):
        # Create padded sequences
        padded_sequences = [pad_sequence(input_d[j][0:i], words_per_phrase) for i in range(1, input_d.shape[1] + 1)]
        input_translation.append(padded_sequences)
    return jnp.array(input_translation)


def cross_entropy_loss(predictions, target):
    # Cross-entropy loss for a batch of predictions and targets
    batch_loss = -jnp.sum(target * jnp.log(predictions + 1e-9), axis=1)
    return jnp.mean(batch_loss)


def diff_norm(X, var, mu, N):
    epsilon = 1e-6
    AA = ((1 - (1 / N)) * (1 / (jnp.sqrt(var + epsilon))))
    BB = (1 / N) * ((X - mu) ** 2)
    CC = ((var + epsilon) ** (3 / 2))
    result = (AA - (BB / CC))
    return result


def redimension(X):
    return jnp.concatenate(jnp.swapaxes(X, 0, 1), axis=-1)


def diffQKV(dAttention, Attention_weights, X1, X2, X3, dk, matrix=""):
    global num_heads
    # print("----------------")
    # print("dAcr.shape",dAttention.shape)
    dAttention_weights = Attention_weights * (1 - Attention_weights)
    # print("dW.shape",dAttention_weights.shape)
    # print("QC.shape",X1.shape)
    # print("VC.shape",X2.shape)
    # print("Ecout.shape",X3.shape)

    if matrix != "k":
        dLoss_dX = jnp.transpose(dAttention, (0, 2, 1)) @ (
                    redimension(dAttention_weights @ (X1 * X2) / jnp.sqrt(dk)) * X3)
    else:
        #  print("X1@jnp.transpose(X2,(0,2,1))",(dAttention_weights*(X1@jnp.transpose(X2,(0,1,3,2)))).shape)
        #  print(X3[0].shape)
        ################################ check X3
        X3 = jnp.swapaxes(jnp.array(jnp.array_split(X3, num_heads, axis=2)), 0, 1)
        #  print(X3[0].shape)
        dLoss_dX = jnp.transpose(dAttention, (0, 2, 1)) @ redimension(
            (dAttention_weights * (X1 @ jnp.transpose(X2, (0, 1, 3, 2))) @ X3) / jnp.sqrt(dk))
    # print("dLoss_dX.shape",dLoss_dX.shape)

    # V1=dAttention_weights@X1/jnp.sqrt(dk)
    # print("V1 shape",V1.shape)
    # V2=jnp.transpose(X2,(0,1,3,2))
    # print("V2 shape",V2.shape)
    # print("(V1@V2)",(V1@V2).shape)
    # V3=redimension(V1@V2)*X3

    # V3=V1@jnp.transpose(V2,(0,2,1))*X3

    return dLoss_dX


@log_time
def create_vocabulary(complete_text, name, nlp):
    # Use re.findall to split considering punctuation
    text = re.findall(r'\[.*?\]|\w+|[^\w\s]|\n', complete_text)

    words_list = list(set(text))

    vocabulary = dict()

    for i, j in enumerate(words_list):
        # vocabulary[j]=(jax.random.uniform(jax.random.key(np.random.randint(10000)),embedding_size),i)
        vocabulary[j] = (jnp.array(nlp(j).vector), i)
        # print(j,len(jnp.array(nlp(j).vector)))

    # print(vocabulary)
    # print("Vocabulary size: ", len(vocabulary))
    with open(f"data/{name}.pkl", 'wb') as handle:
        pickle.dump(vocabulary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return vocabulary


@log_time
def pad_sequences(sentences, lenght, pad_token='[PAD]', target_type=None):
    """
    Pads the input sentences to have the same length by adding [PAD] tokens at the end.
    """

    if target_type == "encoder":
        # Split each sentence into words
        tokenized_sentences = [["[START]"] + re.findall(r'\[.*?\]|\w+|[^\w\s]\n', sentence) + ["[END]"] for sentence in
                               sentences]
    elif target_type == "decoder":
        tokenized_sentences = [["[START]"] + re.findall(r'\[.*?\]|\w+|[^\w\s]|\n', sentence) for sentence in sentences]
    elif target_type == "target":
        tokenized_sentences = [re.findall(r'\[.*?\]|\w+|[^\w\s]|\n', sentence) + ["[END]"] for sentence in sentences]
    # print(tokenized_sentences)
    if lenght == 0:
        # Find the maximum sentence length
        max_len = max(len(sentence) for sentence in tokenized_sentences)
    else:
        max_len = lenght

    # Pad each sentence with the [PAD] token to make them of equal length
    padded_sentences = [" ".join(sentence + [pad_token] * (max_len - len(sentence))) for sentence in
                        tokenized_sentences]

    return padded_sentences


@log_time
def generate_input_encoder(x_batch, vocabulary_encoder, max_words_per_phrase):
    x_train = pad_sequences(x_batch, max_words_per_phrase, target_type="encoder")

    xi = []
    # print(x_batch)
    phrase_vectors_x = [re.findall(r'\[.*?\]|\w+|[^\w\s]|\n', x) for x in x_train]

    phrase_vectors_x = [i[0:max_words_per_phrase] for i in phrase_vectors_x]

    print(phrase_vectors_x)
    # a=np.array(phrase_vectors_x).shape

    # print("a",a)
    # print("len phrases:\n",[len(i) for i in phrase_vectors_x])

    xi = jnp.array([[vocabulary_encoder[word][0] for word in phrase_vector] for phrase_vector in phrase_vectors_x])

    return xi


# @log_time
def update_wembedding_encoder(x_batch, inputs_e, vocabulary, max_words_per_phrase):
    x_train = pad_sequences(x_batch, max_words_per_phrase, target_type="encoder")
    # print(x_train)

    phrase_vectors_x = [re.findall(r'\[.*?\]|\w+|[^\w\s]|\n', x) for x in x_train]

    phrase_vectors_x = [i[0:max_words_per_phrase] for i in phrase_vectors_x]
    # print("inputs_e",inputs_e.shape)
    # print("(phrase_vectors_x).shape:\n",jnp.array(phrase_vectors_x).shape)

    for phrase in range(inputs_e.shape[0]):
        # print(phrase)
        for position, word in enumerate(phrase_vectors_x[phrase]):
            #   print("word",word)
            #   print("original values",vocabulary[word][0])
            #   print("updated  values",inputs_e[phrase][position])
            #   print("index",vocabulary[word][1])
            vocabulary[word] = (inputs_e[phrase][position], vocabulary[word][1])

    return vocabulary


# @log_time
def update_wembedding_decoder(y_batch, inputs_decoder, max_words_per_phrase, vocabulary):
    # print("inputs_decoder",inputs_decoder.shape)
    decoder_input = pad_sequences(y_batch, lenght=max_words_per_phrase, target_type="decoder")
    decoder_input = [i.split() for i in decoder_input]
    # print(max_words_per_phrase)
    if max_words_per_phrase == None:
        max_words_per_phrase = len(decoder_input[0])

    phrase_vectors_y = [i[0:max_words_per_phrase] for i in decoder_input]
    # for sentence in phrase_vectors_y:
    #     print(sentence)
    for phrase in range(inputs_decoder.shape[0]):
        # print(phrase)
        for position, word in enumerate(phrase_vectors_y[phrase]):
            # print("word",word)
            # print("original values",vocabulary[word][0])
            # print("updated  values",inputs_decoder[phrase][position])
            #  print("delta input",np.sum(vocabulary[word][0]-inputs_decoder[phrase][position]))
            # print("index",vocabulary[word][1])
            vocabulary[word] = (inputs_decoder[phrase][position], vocabulary[word][1])

    return vocabulary


@log_time
def create_input_encoder(X, vocabulary_encoder, max_words_per_phrase, embedding_size):
    pos_encoding = get_positional_encoding(max_words_per_phrase, embedding_size)
    inputs_e = generate_input_encoder(X, vocabulary_encoder, max_words_per_phrase)
    inputs_e += pos_encoding
    return inputs_e


@log_time
def create_decoder_input(y_train, embedding_size, max_words_per_phrase, vocabulary_decoder):
    decoder_input = pad_sequences(y_train, lenght=max_words_per_phrase, target_type="decoder")
    decoder_input = [i.split() for i in decoder_input]
    # print(max_words_per_phrase)
    if max_words_per_phrase == None:
        max_words_per_phrase = len(decoder_input[0])

    phrase_vectors_y = [i[0:max_words_per_phrase] for i in decoder_input]
    # for sentence in phrase_vectors_y:
    #     print(sentence)
    yi = jnp.array([[vocabulary_decoder[word][0] for word in phrase_vector] for phrase_vector in phrase_vectors_y])
    pos_encoding = get_positional_encoding(max_words_per_phrase, embedding_size)
    # print(pos_encoding.shape,yi.shape)
    yi = yi + pos_encoding

    decoder_inputs = np.array(np.swapaxes(create_timestaped_input(yi, max_words_per_phrase), 0, 1))

    # decoder_inputs[zero_rows] = vocabulary_decoder["[PAD]"][0]
    for i in range(decoder_inputs.shape[0]):
        for j in range(decoder_inputs[i].shape[0]):
            zero_rows = np.all(decoder_inputs[i][j] == 0, axis=1)

            decoder_inputs[i][j][zero_rows] = vocabulary_decoder["[PAD]"][0]

    decoder_inputs = jnp.array(decoder_inputs)
    return decoder_inputs


# @log_time
def get_one_hot(word, vocabulary_decoder):
    # print(word)
    vocab_size = len(vocabulary_decoder)
    one_hot_vector = np.zeros(vocab_size)
    one_hot_vector[vocabulary_decoder[word][1]] = 1
    return one_hot_vector


@log_time
def create_target(y_train, max_words_per_phrase, vocabulary):
    target_d = pad_sequences(y_train, lenght=max_words_per_phrase, target_type="target")
    target_d = [i.split() for i in target_d]
    target_d = [[get_one_hot(word, vocabulary) for word in phrase] for phrase in target_d]
    # print([len(i) for i in target_d])
    target_d = jnp.swapaxes(create_timestaped_input(jnp.array(target_d), max_words_per_phrase), 0, 1)
    targets_d = []

    for i in range(target_d.shape[0]):
        ff = []
        # print(i,target_d[i].shape)
        for j in range(target_d[i].shape[0]):
            ff.append(target_d[i][j][i])
            # print(targets_d[i][j][i])
        targets_d.append(ff)
        # print(ff)
    targets_d = jnp.array(targets_d)
    return targets_d


def bucket_by_length(x_train, y_train, batch_size):
    # Pair x_train and y_train together
    paired_data = list(zip(x_train, y_train))

    # Sort the pairs by the length of the x_train phrases
    paired_data_sorted = sorted(paired_data, key=lambda pair: len(pair[1]))

    # Initialize buckets for x_train and y_train
    x_batches = []
    y_batches = []

    # Group phrases into batches of batch_size
    for i in range(0, len(paired_data_sorted), batch_size):
        batch = paired_data_sorted[i:i + batch_size]

        # Split the batch back into x_train and y_train
        x_batch, y_batch = zip(*batch)

        x_batches.append(list(x_batch))
        y_batches.append(list(y_batch))

    return x_batches, y_batches


def pad_inputs(sentences, target_type):
    if target_type == "encoder":
        tokenized_sentences = [['[START]'] + sentence + ['[END]'] for sentence in sentences]
    elif target_type == "decoder":
        tokenized_sentences = [["[START]"] + sentence for sentence in sentences]
    elif target_type == "target":
        tokenized_sentences = [sentence + ["[END]"] for sentence in sentences]
    return tokenized_sentences


def generate_inputs(sentences, vocabulary, pad_token='[PAD]', target_type=None):
    """
    Pads the input sentences to have the same length by adding [PAD] tokens at the end.
    """

    tokenized_sentences = pad_inputs(sentences, target_type)
    max_len = max(len(sentence) for sentence in tokenized_sentences)
    # Pad each sentence with the [PAD] token to make them of equal length
    padded_sentences = [sentence + [pad_token] * (max_len - len(sentence)) for sentence in tokenized_sentences]

    # print(padded_sentences)
    xi = jnp.array([[vocabulary[word][0] for word in phrase_vector] for phrase_vector in padded_sentences])
    # print(xi.shape)
    pos_encoding = get_positional_encoding(max_len, 300)
    xi += pos_encoding

    if target_type == "decoder":
        xi = np.array(np.swapaxes(create_timestaped_input(xi, max_len), 0, 1))

    return xi


def forward_attention_encoder(inputs_e):
    global Qe, Ke, Ve, num_heads, batch_size, dk

    Q_E = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(inputs_e, Qe), num_heads, axis=2)), 0, 1)
    # print("Qval.shape: ",Q_E.shape)

    K_E = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(inputs_e, Ke), num_heads, axis=2)), 0, 1)
    # print("Kval.shape: ",K_E.shape)

    V_E = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(inputs_e, Ve), num_heads, axis=2)), 0, 1)
    # print("Vval.shape: ",V_E.shape)

    QKscaled = jnp.matmul(Q_E, jnp.transpose(K_E, (0, 1, 3, 2))) / jnp.sqrt(dk)

    Attention_weights_e = softmax(QKscaled)
    # print("Attention_weights shape:",Attention_weights_e.shape)

    Ae = jnp.matmul(Attention_weights_e, V_E)
    # print("Attention shape:",Ae.shape)

    Ae = jnp.array([jnp.concatenate(Ae[i], axis=1) for i in range(batch_size)])
    # print("Attention encoder shape concat:",Ae.shape)

    return Ae, Attention_weights_e, K_E, V_E, Q_E


def encoder_first_residual_and_norm(Ae, inputs_e):
    Xe = Ae + inputs_e
    Ect1, mu_e, var_e, Ne = layer_norm(Xe)
    # print("Ect1.shape",Ect1.shape,Ne)
    return Ect1, Xe, mu_e, var_e, Ne


def fully_connected_layers_encoder(Ect1):
    global Wfl1e, bfl1e, Wfl2e, bfl2e

    Xe1 = jnp.matmul(Ect1, Wfl1e) + bfl1e
    FLe1 = relu(Xe1)

    FLe2 = jnp.matmul(FLe1, Wfl2e) + bfl2e

    Xe2 = FLe2 + Ect1
    Ecout, mu_e2, var_e2, N_e2 = layer_norm(Xe2)
    # print("Ecout.shape",Ecout.shape)
    return Ecout, mu_e2, var_e2, N_e2, FLe1, Xe1, Xe2


def cross_attention_encoder(Ecout):
    global Kc, Vc
    K_C = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(Ecout, Kc), num_heads, axis=2)), 0, 1)
    # print("K_C.shape: ",K_C.shape)# shape is: num_phrases, numbheads, words_per_phrase, dv/num_heads
    V_C = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(Ecout, Vc), num_heads, axis=2)), 0, 1)
    # print("V_C.shape: ",V_C.shape)
    return K_C, V_C


########################################################DECODER
def forward_attention_decoder(input_decoder):
    global Qd, Kd, Vd  # ,words_per_phrase

    Q_D = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(input_decoder, Qd), num_heads, axis=2)), 0, 1)
    # print("Qval.shape: ",Q_D.shape)# numwords, num_phrases, numheads, num_words, dv/num_heads

    # K_D  = jnp.swapaxes(jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(inputs_d[step], Kd),num_heads,axis=3)), 0, 1),1,2)
    K_D = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(input_decoder, Kd), num_heads, axis=2)), 0, 1)
    # print("Kval.shape: ",K_D.shape)

    # V_D  = jnp.swapaxes(jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(inputs_d[step], Vd),num_heads,axis=3)), 0, 1),1,2)
    V_D = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(input_decoder, Vd), num_heads, axis=2)), 0, 1)

    QKscaled_decoder = jnp.matmul(Q_D, jnp.transpose(K_D, (0, 1, 3, 2))) / jnp.sqrt(dv)
    # Step 1: Create a causal mask of shape (1, 1, 9, 9) to broadcast across heads and batch
    mask_size = input_decoder.shape[1]  # words_per_phrase
    # print(mask_size)
    mask = jnp.tril(jnp.ones((mask_size, mask_size)))  # (9, 9) lower triangular matrix
    mask = mask.at[mask == 0].set(-jnp.inf)  # Set future tokens to -inf
    mask = mask.at[mask == 1].set(0)  # Set allowed tokens to 0
    mask = mask.reshape(1, 1, mask_size, mask_size)

    # Step 2: Apply mask to QKscaled_decoder (it will broadcast across batch and heads)
    QKscaled_decoder = QKscaled_decoder + mask

    Attention_weights_masked = softmax(QKscaled_decoder)

    A_mask = jnp.matmul(Attention_weights_masked, V_D)
    # print("A_mask.shape non concat: ",A_mask.shape)

    # A_mask=jnp.swapaxes(jnp.concatenate(jnp.swapaxes(A_mask,0,2),axis=-1),0,1)
    A_mask = jnp.concatenate(jnp.swapaxes(A_mask, 0, 1), axis=-1)

    # print("A_mask.shape concat: ",A_mask.shape)
    # print("inputs_d.shape: ",input_decoder.shape)
    # print("Dt1.shape: ",Dt1.shape)
    return A_mask, Attention_weights_masked, Q_D, K_D, V_D


def decoder_first_residual_and_norm(A_mask, input_decoder):
    Xd = input_decoder + A_mask
    Dt1, mu_d, var_d, N_d = layer_norm(Xd)
    return Xd, Dt1, mu_d, var_d, N_d


def cross_attention_decoder(Dt1):
    global Qc
    Q_C = jnp.swapaxes(jnp.array(jnp.array_split(jnp.matmul(Dt1, Qc), num_heads, axis=2)), 0, 1)
    # print("Q_C.shape: ",Q_C.shape)
    return Q_C


def cross_attention(Q_C, K_C, V_C, Dt1):
    global dv
    # print("Q_C.shape",Q_C.shape)
    # print("K_C.shape",K_C.shape)
    # print("V_C.shape",V_C.shape)
    QKscaled_cross_attention = jnp.matmul(Q_C, jnp.transpose(K_C, (0, 1, 3, 2))) / jnp.sqrt(dv)
    Attention_weights_cross = softmax(QKscaled_cross_attention)
    Acr = jnp.matmul(Attention_weights_cross, V_C)
    # print("Acr.shape non concat",Acr.shape)
    Acr = jnp.concatenate(jnp.swapaxes(Acr, 0, 1), axis=-1)
    # print("Acr.shape concat",Acr.shape)

    return Acr, Attention_weights_cross


def cross_attention_residual_and_norm(Acr, Dt1):
    Res = Acr + Dt1
    Dt2, mu_res, var_res, N_res = layer_norm(Res)
    return Dt2, mu_res, var_res, N_res, Res


def fully_connected_layers_decoder(Dt2):
    global Wfl1d, bfl1d, Wfl2d, bfl2d, num_phrases

    Xd1 = jnp.matmul(Dt2, Wfl1d) + bfl1d

    FLd1 = relu(Xd1)

    FLd2 = jnp.matmul(FLd1, Wfl2d) + bfl2d
    # print("FLd2.shape",FLd2.shape)

    Xd2 = FLd2 + Dt2
    Dout, mu_d2, var_d2, N_d2 = layer_norm(Xd2)

    # print("Dout.shape",Dout.shape)
    Dout = Dout.reshape(Dout.shape[0], Dout.shape[1] * Dout.shape[2])
    # print("Dout.shape",Dout.shape)
    return Dout, mu_d2, var_d2, N_d2, Xd2, Xd1, FLd1


def output_layer(Dout):
    global Wo, bo

    Zout = jnp.matmul(Dout, Wo) + bo

    SigmaZout = softmax(Zout)
    # print("SigmaZout.shape",SigmaZout.shape)

    return SigmaZout


def loss_calculation(SigmaZout, target):
    # print("target.shape",jnp.array(target).shape)
    Loss = cross_entropy_loss(SigmaZout, target)
    # print("Loss:",Loss)
    return Loss


##################################################################BACKPROPAGATION
def derivate_dout(SigmaZout, target, Dout):
    global Wo, embedding_size, batch_size, words_per_phrase
    dLoss_dZout = SigmaZout - target
    # print("dLoss_dZout.shape",dLoss_dZout.shape)
    dLoss_W0 = jnp.transpose(dLoss_dZout, (1, 0)) @ Dout
    # print("dLoss_W0.shape",dLoss_W0.shape,"W0.shape",W0.shape)
    dLoss_b0 = jnp.sum(dLoss_dZout, axis=0)
    # print("dLoss_b0.shape",dLoss_b0.shape,"b0.shape",b0.shape)
    dLoss_Dout = dLoss_dZout @ Wo.T
    dLoss_Dout = dLoss_Dout.reshape(batch_size, words_per_phrase, embedding_size)
    # print("dLoss_Dout.shape",dLoss_Dout.shape)
    return dLoss_dZout, dLoss_Dout, dLoss_W0, dLoss_b0


def relu_backward(dA, Xd1):
    # Create mask where Xd1 > 0
    relu_mask = (Xd1 > 0).astype(float)  # This will give 1 where Xd1 > 0, else 0
    # Multiply the incoming gradient dA by the mask
    dXd1 = dA * relu_mask
    return dXd1


def derivate_fully_connected_layers_decoder(dLoss_Dout, Dt2, Xd2, var_d2, mu_d2, N_d2, FLd1, Xd1):
    global Wfl1d, Wfl2d

    dLoss_FLd2 = dLoss_Dout * diff_norm(Xd2, var_d2, mu_d2, N_d2)
    # print("dLoss_FLd2.shape",dLoss_FLd2.shape)
    dLoss_Dt2_a = dLoss_FLd2
    # print("dLoss_Dt2_a.shape",dLoss_Dt2_a.shape)
    # print("Dt2.shape",Dt2.shape)
    dLoss_FLd1 = dLoss_FLd2 @ jnp.transpose(Wfl2d, (1, 0))
    # print("dLoss_FLd1.shape",dLoss_FLd1.shape)
    # print("FLd1.shape",FLd1.shape)
    dLoss_Wfl2d = jnp.sum(jnp.transpose(dLoss_FLd2, (0, 2, 1)) @ FLd1, axis=0).T
    # print("dLoss_Wfl2d.shape",dLoss_Wfl2d.shape) # do the mean here over each phrase
    # print("Wfl2d.shape",Wfl2d.shape)
    dLoss_bfl2d = jnp.sum(jnp.sum(dLoss_FLd2, axis=0), axis=0)
    # print("dLoss_bfl2d.shape",dLoss_bfl2d.shape) # do the mean here over each phrase
    # print("bfl2d.shape",bfl2d.shape)

    DLoss_Dt2_b = relu_backward(dLoss_FLd1, Xd1) @ jnp.transpose(Wfl1d, (1, 0))
    DLoss_Dt2 = dLoss_Dt2_a + DLoss_Dt2_b
    # print("DLoss_Dt2.shape",DLoss_Dt2.shape) # do the mean here over each phrase
    # print("Dt2.shape",Dt2.shape)

    dLoss_Wfl1d = jnp.sum(jnp.transpose(relu_backward(dLoss_FLd1, Xd1), (0, 2, 1)) @ Dt2, axis=0).T
    # print("dLoss_Wfl1d.shape",dLoss_Wfl1d.shape) # do the mean here over each phrase
    # print("Wfl1d.shape",Wfl1d.shape)

    dLoss_bfl1d = jnp.sum(jnp.sum(relu_backward(dLoss_FLd1, Xd1), axis=0), axis=0)

    return dLoss_Wfl2d, dLoss_bfl2d, dLoss_Wfl1d, dLoss_bfl1d, DLoss_Dt2


def derivative_cross_attention(dLoss_Dt2, Res, var_res, mu_res, N_res, Attention_weights_cross, K_C, V_C, Q_C, Ecout,
                               Dt1):
    # print("dLoss_bfl1d.shape",dLoss_bfl1d.shape) # do the mean here over each phrase
    # print("bfl1d.shape",bfl1d.shape)
    global dk

    dLoss_Acr = dLoss_Dt2 * diff_norm(Res, var_res, mu_res, N_res)
    # print("dLoss_Acr.shape",dLoss_Acr.shape) # do the mean here over each phrase
    # print("Acr.shape",Acr.shape)
    dLoss_Dt1_a = dLoss_Dt2 * diff_norm(Res, var_res, mu_res, N_res)
    # print("dLoss_Dt1.shape-------",dLoss_Dt1_a.shape) # do the mean here over each phrase
    # print("Dt1.shape",Dt1.shape)

    dLoss_Qc = diffQKV(dLoss_Acr, Attention_weights_cross, K_C, V_C, Dt1, dk)
    # print("dLoss_dQc.shape",dLoss_Qc.shape) # do the mean here over each phrase
    # print("Qc.shape",Qc.shape)
    dLoss_Kc = diffQKV(dLoss_Acr, Attention_weights_cross, Q_C, V_C, Ecout, dk, matrix="k")
    # print("dLoss_dKc.shape",dLoss_Kc.shape) # do the mean here over each phrase
    # print("Kc.shape",Kc.shape)
    dLoss_Vc = np.sum(np.sum(np.transpose(np.expand_dims(dLoss_Acr, axis=1), (0, 1, 3, 2)) @ (
                Attention_weights_cross @ np.expand_dims(Ecout, axis=1)), axis=1), axis=0)
    # print("dLoss_dVc.shape",dLoss_Vc.shape) # do the mean here over each phrase
    # print("Vc.shape",Vc.shape)
    return dLoss_Qc, dLoss_Kc, dLoss_Vc, Attention_weights_cross, dLoss_Dt1_a, dLoss_Acr


def derivative_attention_decoder(dLoss_Acr, Attention_weights_cross, dLoss_Dt1_a, Attention_weights_masked, Q_D, V_D,
                                 K_D, K_C, V_C, Xd, var_d, mu_d, N_d, input_d):
    global Qc, dk

    dAttention_weights_cross = Attention_weights_cross * (1 - Attention_weights_cross)
    # print("dAttention_weights_cross.shape",dAttention_weights_cross.shape)
    # V1=redimension(dAttention_weights_cross@K_C/jnp.sqrt(dk))
    # print("K_C.shape",K_C.shape)
    # V2=redimension(V_C)
    # print("V_C.shape",V_C.shape)
    # print("Qc.shape",Qc.shape)

    # print("dLoss_Dt1_b.shape",dLoss_Dt1_a.shape)
    # V3=V1*V2@Qc
    # dLoss_Dt1_b=dLoss_Acr*V3
    # V3=V1@jnp.transpose(V2@Qc,(0,2,1))
    # dLoss_Dt1_b=jnp.transpose(dLoss_Acr,(0,2,1))@V3

    V1 = dAttention_weights_cross
    V2 = K_C * V_C / jnp.sqrt(dk)
    V3 = redimension(V1 @ V2) @ Qc

    # print("V3.shape",dLoss_Dt1_a.shape)
    # print("dLoss_Acr.shape",dLoss_Acr.shape)
    dLoss_Dt1_b = dLoss_Acr * V3

    dLoss_Dt1 = dLoss_Dt1_a + dLoss_Dt1_b

    dLoss_Amask = dLoss_Dt1 * diff_norm(Xd, var_d, mu_d, N_d)
    # print("dLoss_DAmask.shape",dLoss_Amask.shape)
    dLoss_inputd_a = dLoss_Amask
    # print("dLoss_Dinputd_a.shape",dLoss_inputd_a.shape)
    dLoss_Kd = diffQKV(dLoss_Amask, Attention_weights_masked, Q_D, V_D, input_d, dk)
    # print("dLoss_Kd.shape",dLoss_Kd.shape)
    dLoss_Qd = diffQKV(dLoss_Amask, Attention_weights_masked, K_D, V_D, input_d, dk)
    # print("dLoss_Qd.shape",dLoss_Qd.shape)
    dLoss_Vd = np.sum(np.sum(np.transpose(np.expand_dims(dLoss_Amask, axis=1), (0, 1, 3, 2)) @ (
                Attention_weights_masked @ np.expand_dims(input_d, axis=1)), axis=1), axis=0)
    return dLoss_Kd, dLoss_Qd, dLoss_Vd, dLoss_inputd_a, dLoss_Amask


def derivative_input_decoder(dLoss_Amask, Attention_weights_masked, K_D, V_D, Q_D, dLoss_inputd_a, input_d):
    global Qd, Kd, Vd, dk
    dLoss_V_D = np.transpose(
        np.sum(np.transpose(np.expand_dims(dLoss_Amask, axis=1), (0, 1, 3, 2)) @ Attention_weights_masked, axis=1),
        (0, 2, 1))
    dLoss_V_D.shape
    dLoss_inputd_v = dLoss_V_D @ Vd

    # print("dLoss_inputd_v.shape",dLoss_inputd_v.shape) # do the mean here over each phrase
    # print("input_d.shape",input_d.shape)

    dAttention_weights_masked = Attention_weights_masked * (1 - Attention_weights_masked)
    V1 = redimension(dAttention_weights_masked @ K_D / jnp.sqrt(dk))
    V2 = redimension(V_D)
    V3 = V1 * V2
    dLoss_Q_D = dLoss_Amask * V3

    dLoss_inputd_q = dLoss_Q_D @ Qd
    # print("dLoss_inputd_q.shape",dLoss_inputd_q.shape)

    V1 = redimension(dAttention_weights_masked @ Q_D / jnp.sqrt(dk))
    V2 = redimension(V_D)
    V3 = V1 * V2
    dLoss_K_D = dLoss_Amask * V3
    dLoss_inputd_k = dLoss_K_D @ Kd
    # print("dLoss_inputd_k.shape",dLoss_inputd_k.shape)
    dLoss_inputd = dLoss_inputd_a + dLoss_inputd_k + dLoss_inputd_q + dLoss_inputd_v

    dLoss_dWemb_decoder = dLoss_inputd * input_d

    return dLoss_inputd, dLoss_dWemb_decoder


def derivative_Ecout(Attention_weights_cross, dLoss_Acr, Q_C, V_C):
    global Kc, Vc, num_heads

    dAttention_weights_cross = Attention_weights_cross * (1 - Attention_weights_cross)
    # dLoss_Acr=jnp.expand_dims(dLoss_Acr,axis=1)
    # print("--------------")
    # print("dLoss_Acr.shape",dLoss_Acr.shape)
    # print("dAttention_weights_cross.shape",dAttention_weights_cross.shape)
    # print("Q_C.shape",Q_C.shape)
    # print("V_C.shape",V_C.shape)
    # print("Kc.shape",Kc.shape)
    # V1=redimension(dAttention_weights_cross@Q_C/jnp.sqrt(dk))

    # V2=redimension(V_C)

    # V3=V1*V2

    # dLoss_K_C=dLoss_Acr*V3
    dLoss_Acr = jnp.swapaxes(jnp.array(jnp.array_split(dLoss_Acr, num_heads, axis=2)), 0, 1)
    V1 = Q_C @ jnp.transpose(V_C, (0, 1, 3, 2)) / jnp.sqrt(dk)
    # print("V1.shape",V1.shape)
    V2 = dAttention_weights_cross * V1
    # print("V2.shape",V2.shape,"dLoss_Acr.shape",dLoss_Acr.shape)
    dLoss_K_C = jnp.transpose(jnp.transpose(dLoss_Acr, (0, 1, 3, 2)) @ V2, (0, 1, 3, 2))
    # print("dLoss_K_C.shape",dLoss_K_C.shape)

    dLoss_Ecout_k = redimension(dLoss_K_C) @ Kc
    # print("dLoss_Ecout_k.shape",dLoss_Ecout_k.shape)
    # dLoss_K_C=dLoss_Acr@jnp.transpose(redimension(V2),(0,2,1))
    # print("dLoss_K_C.shape",dLoss_K_C.shape)

    # dLoss_Ecout_k=dLoss_K_C@Kc
    # print("dLoss_Ecout_k.shape",dLoss_Ecout_k.shape)
    dLoss_V_C = jnp.transpose(jnp.transpose(dLoss_Acr, (0, 1, 3, 2)) @ Attention_weights_cross, (0, 1, 3, 2))
    # print("dLoss_V_C.shape",dLoss_V_C.shape)
    # dLoss_V_C=np.transpose(np.sum(np.transpose(np.expand_dims(dLoss_Acr, axis=1),(0,1,3,2))@Attention_weights_cross,axis=1),(0,2,1))
    # dLoss_V_C.shape
    dLoss_Ecout_v = redimension(dLoss_V_C) @ Vc
    # print("dLoss_Ecout_v.shape",dLoss_Ecout_v.shape)
    # print("dLoss_Ecout_v.shape",dLoss_Ecout_v.shape) # do the mean here over each phrase
    dLoss_Ecout = dLoss_Ecout_k + dLoss_Ecout_v
    return dLoss_Ecout


def derivate_fully_connected_layers_encoder(dLoss_Ecout, Ect1, Xe2, var_e2, mu_e2, N_e2, FLe1, Xe1):
    global Wfl2e, Wfl1e
    dLoss_dFLe2 = dLoss_Ecout * diff_norm(Xe2, var_e2, mu_e2, N_e2)
    dLoss_Ect1_a = dLoss_dFLe2
    # print(Wfl2e.shape)
    dLoss_dFLe1 = dLoss_dFLe2 @ jnp.transpose(Wfl2e, (1, 0))
    dLoss_dWfl2e = jnp.transpose(dLoss_dFLe2, (0, 2, 1)) @ FLe1
    # print(dLoss_dWfl2e)
    dLoss_dbfl2e = jnp.sum(dLoss_dFLe2, axis=1)

    dLoss_Ect1_b = relu_backward(dLoss_dFLe1, Xe1) @ jnp.transpose(Wfl1e, (1, 0))

    dLoss_Ect1 = dLoss_Ect1_b + dLoss_Ect1_a

    dLoss_Wfl1e = jnp.transpose(relu_backward(dLoss_dFLe1, Xe1), (0, 2, 1)) @ Ect1

    dLoss_bfl1e = jnp.transpose(relu_backward(dLoss_dFLe1, Xe1), (0, 2, 1))

    return dLoss_dWfl2e, dLoss_dbfl2e, dLoss_Wfl1e, dLoss_bfl1e, dLoss_Ect1


import warnings


def derivative_attention_encoder(dLoss_Ect1, Xe, var_e, mu_e, Ne, Attention_weights_e, K_E, V_E, Q_E, inputs_e):
    global dk
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        try:
            dLoss_Ae = dLoss_Ect1 * diff_norm(Xe, var_e, mu_e, Ne)

            dLoss_inpute_a = dLoss_Ae

            dLoss_dQe = diffQKV(dLoss_Ae, Attention_weights_e, K_E, V_E, inputs_e, dk)
            # print("dLoss_dQe.shape",dLoss_dQe.shape) # do the mean here over each phrase
            # print("Qe.shape",Qe.shape)
            dLoss_dKe = diffQKV(dLoss_Ae, Attention_weights_e, Q_E, V_E, inputs_e, dk)
            # print("dLoss_dKe.shape",dLoss_dKe.shape) # do the mean here over each phrase
            # print("Ke.shape",Ke.shape)

            dLoss_dVe = np.sum(np.sum(np.transpose(np.expand_dims(dLoss_Ae, axis=1), (0, 1, 3, 2)) @ (
                        Attention_weights_e @ np.expand_dims(inputs_e, axis=1)), axis=1), axis=0)
            # print("dLoss_dVe.shape",dLoss_dVe.shape) # do the mean here over each phrase
            return dLoss_dQe, dLoss_dKe, dLoss_dVe, dLoss_inpute_a, dLoss_Ae

        except RuntimeWarning as rw:
            # Check for NaN or inf values in inputs and matrices
            # print("dLoss_Ae check ", dLoss_Ae)
            # print("Attention_weights_e  ", Attention_weights_e)
            # print("inputs_e  ", inputs_e)
            print(f"Caught a RuntimeWarning: {rw}")
            return None  # Return None if a warning occurs

        except Exception as e:
            # Additional checks in case of other exceptions
            # print("inputs_e check ", np.isnan(inputs_e).any(), np.isinf(inputs_e).any())
            # print("Attention_weights_e check ", np.isnan(Attention_weights_e).any(), np.isinf(Attention_weights_e).any())
            # print("dLoss_Ae check ", np.isnan(dLoss_Ae).any(), np.isinf(dLoss_Ae).any())
            # print(f"Caught an error: {e}")
            return None


def derivative_input_encoder(dLoss_Ae, Attention_weights_e, K_E, V_E, Q_E, dLoss_inpute_a, inputs_e):
    global Ve, Qe, Ke, dk

    dLoss_V_E = np.transpose(
        np.mean(np.transpose(np.expand_dims(dLoss_Ae, axis=1), (0, 1, 3, 2)) @ Attention_weights_e, axis=1), (0, 2, 1))
    dLoss_inpute_v = dLoss_V_E @ Ve

    dAttention_weights_e = Attention_weights_e * (1 - Attention_weights_e)
    V1 = redimension(dAttention_weights_e @ K_E / jnp.sqrt(dk))
    V2 = redimension(V_E)
    V3 = V1 * V2
    dLoss_Q_E = dLoss_Ae * V3
    dLoss_inpute_q = dLoss_Q_E @ Qe
    # print("dLoss_inpute_q.shape",dLoss_inpute_q.shape)

    V1 = redimension(dAttention_weights_e @ Q_E / jnp.sqrt(dk))
    V2 = redimension(V_E)
    V3 = V1 * V2
    dLoss_K_E = dLoss_Ae * V3
    dLoss_inpute_k = dLoss_K_E @ Ke
    # print("dLoss_inpute_k.shape",dLoss_inpute_k.shape)
    dLoss_inpute = dLoss_inpute_a + dLoss_inpute_k + dLoss_inpute_q + dLoss_inpute_v
    dLoss_dWemb_encoder = dLoss_inpute * inputs_e
    return dLoss_inpute, dLoss_dWemb_encoder


def print_vocabs(ans, vocabulary):
    for idx, values in enumerate(ans):
        max_index = np.argmax(values)

        # Step 2: Find the word in the vocabulary with the corresponding position
        matched_word = None
        for word, (_, position) in vocabulary.items():
            if position == max_index:
                matched_word = word
                break
        print(f"List {idx + 1}: Max value index: {max_index}, Matched word: {matched_word}")


