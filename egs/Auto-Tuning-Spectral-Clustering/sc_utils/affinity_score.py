from time import time
import math
from pathlib import Path
import pickle
import pprint
import sys
import kaldi_io
import argparse
# import ipdb
import copy
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# Pretty Printer Object for printing with indenting
pp = pprint.PrettyPrinter(indent=4)

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
# decorator fuction for calculating runtime

def write_txt(w_path, list_to_wr):
    with open(w_path, "w") as output:
        for k, val in enumerate(list_to_wr):
            output.write(val + '\n')
    return None

def read_txt(list_path):
    with open(list_path) as f:
        content = []
        for line in f:
            line = line.strip()
            content.append(line)
        f.close()
        assert content != [], "File is empty. Abort. Given path: " + list_path
        return content


def spk2utt2dict(path):
    t_list = open(path)
    out_dict = {}
    for line in t_list:
        key = line.strip().split()[0]
        val = line.strip().split()[1:]
        # scaler.fit(sim_d)
        # sim_d = scaler.transform(sim_d)
        out_dict[key] = val
    return out_dict 

def scp2dict(path):
    t_list = open(path)
    out_dict = {}
    for line in t_list:
        key = line.strip().split()[0]
        val = line.strip().split()[1]
        if key not in out_dict:
            out_dict[key] = val

    return out_dict 

def segment_to_line_dict(segments_path):
    segments_line_list = read_txt(segments_path)
    seg_line_dict = {}
    for line in segments_line_list:
        seg_line_dict[line.split()[0]] = line.strip()
    return seg_line_dict


def get_embed_dict(scp):
    embedding_dict, feats_scp_segments_line_dict = {}, {}
    for seg_id, val in kaldi_io.read_vec_flt_scp(scp):
        # ipdb.set_trace()
        sess_id = utt2spk_dict[seg_id]
        if sess_id not in embedding_dict:
            embedding_dict[sess_id] = [val] 
            feats_scp_segments_line_dict[sess_id] = [seg_line_dict[seg_id]]

        else:
            embedding_dict[sess_id].append(val)
            feats_scp_segments_line_dict[sess_id].append(seg_line_dict[seg_id])
    return embedding_dict, feats_scp_segments_line_dict

def min_var(input_list):
    inlist = []
    for i, mat in enumerate(input_list):
        inlist.append(mat.flatten())

    if type(inlist) == list:
        n_fts = len(inlist) 
        X = np.zeros((len(inlist[0]), len(inlist)))
        for k in range(len(inlist)):
            X[:, k] =  inlist[k]

    om = np.ones((len(inlist),1))
    cov_mat = np.cov(X.T) 
    inv_com = np.linalg.inv(cov_mat) 
    p_mat = np.matmul(inv_com,om)/np.matmul(np.matmul(om.T,inv_com), om ) 
    p_mat = p_mat.T

    # ipdb.set_trace()
    # p_mat = np.expand_dims(ptf_m, axis = 0)
    return p_mat[0][0]

def save_npy(sim_out_dict, out_dir):
    key_list = []
    for key, v in sim_out_dict.items():
        np.save('{}/{}'.format(out_dir, key+'.npy'), v) 
        key_list.append(key)
    write_txt('{}/{}'.format(out_dir, 'scores.txt'), key_list)
    
def p_pruning(A, pval):


    n_elems = int((1 - pval) * A.shape[0])

    # For each row in a affinity matrix
    for i in range(A.shape[0]):
        low_indexes = np.argsort(A[i, :])
        low_indexes = low_indexes[0:n_elems]

        # Replace smaller similarity values by 0s
        A[i, low_indexes] = 0

    return A    
def diagonal_fill(A):
    """
    Sets the diagonal elemnts of the matrix to the max of each row
    """
    np.fill_diagonal(A, 0.0)
    A[np.diag_indices(A.shape[0])] = np.max(A, axis=1)
    return A
from scipy.ndimage import gaussian_filter
def gaussian_blur(A, sigma=1.0):
    """
    Does a gaussian blur on the affinity matrix
    """
    return gaussian_filter(A, sigma=sigma)

def row_threshold_mult(A, p=0.95, mult=0.01):
    """
    For each row multiply elements smaller than the row's p'th percentile by mult
    """
    percentiles = np.percentile(A, p*100, axis=1)
    mask = A < percentiles[:,np.newaxis]

    A = (mask * mult * A) +  (~mask * A)
    return A

def symmetrization(A):
    """
    Symmeterization: Y_{i,j} = max(S_{ij}, S_{ji})
    """
    return np.maximum(A, A.T)

def diffusion(A):
    """
    Diffusion: Y <- YY^T
    """
    return np.dot(A, A.T)

def row_max_norm(A):
    """
    Row-wise max normalization: S_{ij} = Y_{ij} / max_k(Y_{ik})
    """
    maxes = np.amax(A, axis=1)
    return A/maxes

def sim_enhancement(A):
    func_order = [
                  gaussian_blur,
                  diagonal_fill,
                  row_threshold_mult,
                  
                  row_max_norm
                ]
    for f in func_order:
        A = f(A)
    return A

def compute_affinity_matrix(X):
    """Compute the affinity matrix from data.
    Note that the range of affinity is [0,1].
    Args:
        X: numpy array of shape (n_samples, n_features)
    Returns:
        affinity: numpy array of shape (n_samples, n_samples)
    """
    # Normalize the data.
    l2_norms = np.linalg.norm(X, axis=1)
    X_normalized = X / l2_norms[:, None]
    # Compute cosine similarities. Range is [-1,1].
    cosine_similarities = np.matmul(X_normalized, np.transpose(X_normalized))
    # Compute the affinity. Range is [0,1].
    # Note that this step is not mentioned in the paper!
    affinity = (cosine_similarities + 1.0) / 2.0
    return affinity
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', action='store', type=float, default=None)
    parser.add_argument('--read_costs', action='store', type=str)
    parser.add_argument('--reco2num_sess_id_rspecifier', action='store', type=str, default='None')
    parser.add_argument('--scp', action='store', type=str, default=None)
    parser.add_argument('--scp_e2', action='store', type=str, default=None)
    parser.add_argument('--spk2utt', action='store', type=str, default=None)
    parser.add_argument('--utt2spk', action='store', type=str, default=None)
    parser.add_argument('--segments', action='store', type=str, default=None)
    parser.add_argument('--scores', action='store', type=str, default=None)
    parser.add_argument('--parallel_job', action='store', type=str, default='1')
    parser.add_argument('--score_metric', action='store', type=str, default='cos')
    parser.add_argument('--mix_alpha', action='store', type=str, default='None')
    parser.add_argument('--alpha_est_weights', action='store', type=str, default='None')
   
    param = parser.parse_args()
    scaler = MinMaxScaler(feature_range=(0, 1))
    # scaler = MinMaxScaler(feature_range=(-0.5, 0.5))
    
    if "." in param.mix_alpha:
        param.mix_alpha = float(param.mix_alpha)

    scp_dict = scp2dict(param.scp)
    if param.scp_e2 != None:
        scp_dict_e2 = scp2dict(param.scp_e2)
    utt2spk_dict = scp2dict(param.utt2spk)
    spk2utt_dict = spk2utt2dict(param.spk2utt)
    seg_line_dict = segment_to_line_dict(param.segments)
    sess_id_list = list(spk2utt_dict.keys())
    label_out = []
    nC=-1
    if param.alpha_est_weights != 'None':
        cont_alpha_est_weights = read_txt(param.alpha_est_weights)
        alpha_est_weights_dict = { x.split()[0]:float(x.split()[1]) for x in cont_alpha_est_weights }

    if param.score_metric == 'cos':
        embedding_dict, feats_scp_segments_line_dict = get_embed_dict(param.scp)

    if param.score_metric == 'cosFuse':
        embedding_dict, feats_scp_segments_line_dict = {}, {}
        segid2embed_dict = {} 
        for seg_id, val in kaldi_io.read_vec_flt_scp(param.scp_e2):
            segid2embed_dict[seg_id] = val
        
        for seg_id, val in kaldi_io.read_vec_flt_scp(param.scp):

            sess_id = utt2spk_dict[seg_id]
            val_e2 = segid2embed_dict[seg_id]
            val_con = np.hstack((val, val_e2))
            if sess_id not in embedding_dict:
                embedding_dict[sess_id] = [val_con] 
                feats_scp_segments_line_dict[sess_id] = [seg_line_dict[seg_id]]

            else:
                embedding_dict[sess_id].append(val_con)
                feats_scp_segments_line_dict[sess_id].append(seg_line_dict[seg_id])
    
    if param.score_metric in [ 'cosAdd', 'cosWsum']:
        embedding_dict,    feats_scp_segments_line_dict =    get_embed_dict(param.scp)
        embedding_dict_e2, feats_scp_segments_line_dict_e2 = get_embed_dict(param.scp_e2)
    


    sim_out_dict = {}
    output_segment_list = []
    for key, val in embedding_dict.items():
        output_segment_list.extend(feats_scp_segments_line_dict[key])
        embd_mat = np.vstack(embedding_dict[key]) 
        print(embd_mat.shape)
        
        
        # euc_d = euc_dist(embd_mat)
        if param.score_metric == 'euc':
            sim_d = -1*euclidean_distances(embd_mat)
        
        elif param.score_metric in ['cos', 'cosFuse']:
            #sim_d = cosine_similarity(embd_mat)
            sim_d = compute_affinity_matrix(embd_mat)
            sim_d = p_pruning(sim_d, 0.3)
            sim_d = 0.5 * (sim_d + sim_d.T)
            sim_d = sim_enhancement(sim_d)
        elif param.score_metric == 'cosAdd':
            embd_mat_e2 = embedding_dict_e2[key]
            sim_d = cosine_similarity(embd_mat)
            sim_d_e2 = cosine_similarity(embd_mat_e2)
            sim_d = sim_d + sim_d_e2
        
        elif param.score_metric == 'cosWsum':
            embd_mat_e2 = embedding_dict_e2[key]
            sim_d = cosine_similarity(embd_mat)
            sim_d_e2 = cosine_similarity(embd_mat_e2)
            if param.alpha_est_weights != 'None':
                alpha = float(alpha_est_weights_dict[key])
                pass
            else:
                alpha=param.mix_alpha
            print(key, "mix_alpha:", alpha)
            sim_d = alpha*sim_d + (1-alpha)*sim_d_e2

        else:
            raise ValueError("No similarity method")
        scaler.fit(sim_d)
        sim_d = scaler.transform(sim_d)
        
        
        sim_out_dict[key] = sim_d
        assert sim_d.shape[0] == len(val)

    print('Succesfully calculated %d %s similarity'%(len(list(sim_out_dict.keys())), param.score_metric))
    out_dir = '/'.join(param.scores.split('/')[:-1])
    print("out_dir:", out_dir)
    ark_scp_output='ark:| copy-feats --compress=false ark:- ark,scp:' + out_dir + '/scores.' + param.parallel_job + '.ark,' + param.scores
    with kaldi_io.open_or_fd(ark_scp_output, 'wb') as f: 
        for spk_id, mat in sim_out_dict.items():
            kaldi_io.write_mat(f, mat, key=spk_id.rstrip()) 
    
    segments_save_full_path = out_dir + '/' + 'segments.' + param.parallel_job
    write_txt(segments_save_full_path, output_segment_list)

    save_npy(sim_out_dict, out_dir)


