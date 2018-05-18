#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Authors:
# Samuel Läubli <laeubli@cl.uzh.ch>
# Mathias Müller <mmueller@cl.uzh.ch>

import tensorflow as tf

from daikon import constants as C


def compute_lengths(sequences):
    """
    This solution is similar to:
    https://danijar.com/variable-sequence-lengths-in-tensorflow/
    """
    used = tf.sign(tf.abs(sequences))
    lengths = tf.reduce_sum(used, reduction_indices=1)
    lengths = tf.cast(lengths, tf.int32)
    return lengths


def define_computation_graph(source_vocab_size: int, target_vocab_size: int, batch_size: int):

    tf.reset_default_graph()

    # Placeholders for inputs and outputs
    encoder_inputs = tf.placeholder(
        shape=(batch_size, None), dtype=tf.int32, name='encoder_inputs')

    decoder_targets = tf.placeholder(
        shape=(batch_size, None), dtype=tf.int32, name='decoder_targets')
    decoder_inputs = tf.placeholder(
        shape=(batch_size, None), dtype=tf.int32, name='decoder_inputs')

    with tf.variable_scope("Embeddings"):
        source_embedding = tf.get_variable(
            'source_embedding', [source_vocab_size, C.EMBEDDING_SIZE])
        target_embedding = tf.get_variable(
            'target_embedding', [source_vocab_size, C.EMBEDDING_SIZE])

        encoder_inputs_embedded = tf.nn.embedding_lookup(
            source_embedding, encoder_inputs)
        decoder_inputs_embedded = tf.nn.embedding_lookup(
            target_embedding, decoder_inputs)

    with tf.variable_scope("Encoder"):
        encoder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
        initial_state = encoder_cell.zero_state(batch_size, tf.float32)

        encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(encoder_cell,
                                                                 encoder_inputs_embedded,
                                                                 initial_state=initial_state,
                                                                 dtype=tf.float32)
    # with tf.name_scope('modified_Encoder'):
    #
    #     def encoder_cell():
    #         # define GRU cell with dropout
    #         encoder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
    #
    #         return encoder_cell
    #
    #     # stack multiple layers
    #     encoder_block = tf.contrib.rnn.MultiRNNCell(
    #         [encoder_cell() for _ in range(C.NUM_LAYERS)])
    #
    #     initial_state = encoder_block.zero_state(batch_size, tf.float32)
    #
    #     encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    #         encoder_block, encoder_inputs_embedded, initial_state=initial_state, dtype=tf.float32)

    # with tf.name_scope('BiEncoder'):
    #     # define forward cell (use only half of length due to bidirectionality)
    #     encoder_cell_fw = tf.nn.rnn_cell.GRUCell(C.HIDDEN_SIZE/2)
    #
    #     # define backward cell (use only half of length due to bidirectionality)
    #     encoder_cell_bw = tf.nn.rnn_cell.GRUCell(C.HIDDEN_SIZE/2)
    #
    #     initial_state_fw = encoder_cell_fw.zero_state(batch_size, tf.float32)
    #     initial_state_bw = encoder_cell_bw.zero_state(batch_size, tf.float32)
    #
    #     bi_outputs, encoder_final_state = tf.nn.bidirectional_dynamic_rnn(
    #         encoder_cell_fw, encoder_cell_bw, encoder_inputs_embedded, initial_state_fw=initial_state_fw, initial_state_bw=initial_state_bw, dtype=tf.float32)
    #
    #     encoder_outputs = tf.concat(bi_outputs, -1)
    #
    # with tf.name_scope('modified_Decoder'):
    #     def decoder_cell():
    #         # define GRU cell with dropout
    #         decoder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
    #
    #         return decoder_cell
    #
    #     # stack multiple layers
    #     decoder_block = tf.contrib.rnn.MultiRNNCell(
    #         [decoder_cell() for _ in range(C.NUM_LAYERS)])
    #
    #     decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    #         decoder_block, decoder_inputs_embedded, initial_state=encoder_final_state, dtype=tf.float32)

    with tf.variable_scope("Decoder"):
        decoder_cell = tf.contrib.rnn.GRUCell(C.HIDDEN_SIZE)
        decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(decoder_cell,
                                                                 decoder_inputs_embedded,
                                                                 initial_state=encoder_final_state,
                                                                 dtype=tf.float32)

    with tf.variable_scope("Logits"):
        decoder_logits = tf.contrib.layers.linear(
            decoder_outputs, target_vocab_size)

    with tf.variable_scope("Loss"):
        one_hot_labels = tf.one_hot(
            decoder_targets, depth=target_vocab_size, dtype=tf.float32)
        stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=one_hot_labels,
            logits=decoder_logits)

        # mask padded positions
        target_lengths = compute_lengths(decoder_targets)
        target_weights = tf.sequence_mask(
            lengths=target_lengths, maxlen=None, dtype=decoder_logits.dtype)
        weighted_cross_entropy = stepwise_cross_entropy * target_weights
        loss = tf.reduce_mean(weighted_cross_entropy)

    with tf.variable_scope('Optimizer'):
        train_step = tf.train.AdamOptimizer(
            learning_rate=C.LEARNING_RATE).minimize(loss)

    # Logging of cost scalar (@tensorboard)
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

    return encoder_inputs, decoder_targets, decoder_inputs, loss, train_step, decoder_logits, summary
