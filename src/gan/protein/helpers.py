import os
import torch
from torch.nn import functional as F
import numpy as np
#import tensorflow as tf
from gan.parameters import DATASET

ACID_EMBEDDINGS = "acid_embeddings"
ACID_EMBEDDINGS_SCOPE = "acid_emb_scope"
REAL_PROTEINS = "real_proteins"
FAKE_PROTEINS = "fake_proteins"
CLASS_MAPPING = "class_mapping"
LABELS = "labels"
NUM_AMINO_ACIDS = 21
SEQ_LENGTH = "seq_length"


def convert_to_acid_ids(fake_x, batch_size):
    fake_to_display = torch.squeeze(fake_x)
    #acid_embeddings = tf.get_variable(ACID_EMBEDDINGS_SCOPE + "/" + ACID_EMBEDDINGS)
    fake_to_display, distances = reverse_embedding_lookup(acid_embeddings, fake_to_display, batch_size)
    fake_to_display = torch.squeeze(fake_to_display)
    #tf.summary.scalar("FAKE", torch.mean(distances), family="Cosine_distance")
    return fake_to_display, distances


def reverse_embedding_lookup(acid_embeddings, embedded_sequence, batch_size):
    embedded_sequence = torch.transpose(embedded_sequence, [0, 2, 1])
    acid_embeddings_expanded = torch.tile(torch.unsqueeze(acid_embeddings, dim=0), dim=[batch_size, 1, 1])
    emb_distances = torch.matmul(
        F.normalize(acid_embeddings_expanded, p=2, dim=2),
        F.normalize(embedded_sequence, p=2, dim=1))
    indices = torch.argmax(emb_distances, dim=1)
    return indices, torch.max(emb_distances, dim=1)


def test_amino_acid_embeddings(acid_embeddings, real_x, width):
    #print_op_b = tf.print(torch.transpose("REAL_127:", real_x[0], perm=[1, 0])[127, :], summarize=width)
    #print_op_e = tf.print(torch.transpose("REAL_0:", real_x[0], perm=[1, 0])[0, :], summarize=width)
    real_x, _ = reverse_embedding_lookup(acid_embeddings, real_x)
    #print_op_a = tf.print("RECO!: ", real_x[0], summarize=width)


def get_shape(config, properties):
    width = properties[SEQ_LENGTH]
    if config.one_hot:
        return [config.batch_size, 1, width, NUM_AMINO_ACIDS]
    else:
        return [config.batch_size, 1, width, config.embedding_height]


def get_file(filename, flags):
    embedding_path = os.path.join(flags.data_dir, DATASET, filename)
    return np.load(embedding_path)