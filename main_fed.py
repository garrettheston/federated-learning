#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar
from models.Fed import FedAvg
from models.test import test_img

# The NIST Module-Lattice-Based Key-Encapsulation Mechanism Standard ML-KEM 
from kyber_py.ml_kem import ML_KEM_512

# The AES algorithm for the keys actually used after the KEM
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# Import pickle for data serialization
import pickle

# Import hashlib for integrity validation
import hashlib

if __name__ == '__main__':    
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    # copy weights
    w_glob = net_glob.state_dict()

    # training
    loss_train = []
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []
    
    # Hex digests before and after transmission
    pre_transmission_vector = []
    post_transmission_vector = []
    # Validae alteration
    altered = False
    ciphertext = ""
    idxs_users = ""
    
    if args.all_clients: 
        print("Aggregation over all clients")
        w_locals = [w_glob for i in range(args.num_users)]
    for iter in range(args.epochs):
        # Shared_key_receiver session key and initialization vector (iv)
        session_key_and_iv = []
        loss_locals = []
        if not args.all_clients:
            w_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:

            # Generating session keypair
            ek, dk = ML_KEM_512.keygen()
            # Generating shared secret key
            shared_key_sender, ct = ML_KEM_512.encaps(ek)
            # Key decapsulation
            shared_key_receiver = ML_KEM_512.decaps(dk, ct)
            assert shared_key_sender == shared_key_receiver

            iv = get_random_bytes(16)  # Generate random IV

            # Derive cipher for encryption from PEM
            init_cipher = AES.new(shared_key_sender[:32], AES.MODE_OFB, iv)
            
            # Append the receiver key and the iv to the dictionary
            session_key_and_iv.append((shared_key_receiver,iv))
            
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            
            w_serialized = pickle.dumps(w) # serialize the data so that it can be encrypted
            hash_value = hashlib.sha256(w_serialized).hexdigest() # pre-transmission hex digest
            pre_transmission_vector.append(hash_value)
            
            # local model will send updates with ciphertext
            ciphertext = init_cipher.encrypt(w_serialized)
            if args.all_clients:
                w_locals[idx] = copy.deepcopy(ciphertext)
            else:
                w_locals.append(copy.deepcopy(ciphertext))
                
            loss_locals.append(copy.deepcopy(loss))
        
        # update global weights with plaintext
        for i in range(len(w_locals)):
            shared_key_receiver, iv = session_key_and_iv[i]
            new_cipher = AES.new(shared_key_receiver[:32], AES.MODE_OFB, iv)
            decrypted = new_cipher.decrypt(w_locals[i])
            hash_value = hashlib.sha256(decrypted).hexdigest() # post-transmission hex digest
            post_transmission_vector.append(hash_value)
            w_locals[i] = pickle.loads(decrypted)
        
        w_glob = FedAvg(w_locals)

        # copy weight to net_glob
        net_glob.load_state_dict(w_glob)

        # print loss
        loss_avg = sum(loss_locals) / len(loss_locals)
        print('Round {:3d}, Average loss {:.3f}'.format(iter, loss_avg))
        loss_train.append(loss_avg)

    # intrusion detection established by comparing sha256 hex digests before transmission and after transmission
    for i in range(len(w_locals)):
        assert pre_transmission_vector[i] == post_transmission_vector[i]
        
    # plot loss curve
    plt.figure()
    plt.plot(range(len(loss_train)), loss_train)
    plt.ylabel('train_loss')
    plt.savefig('./save/fed_{}_{}_{}_C{}_iid{}.png'.format(args.dataset, args.model, args.epochs, args.frac, args.iid))

    # testing
    net_glob.eval()
    acc_train, loss_train = test_img(net_glob, dataset_train, args)
    acc_test, loss_test = test_img(net_glob, dataset_test, args)
    print("Training accuracy: {:.2f}".format(acc_train))
    print("Testing accuracy: {:.2f}".format(acc_test))



IV:  b'\x9e\x9d\x91\x82\x88\x83-\xf9c\xbao\xef\x99\xd2p\xd7'
IV:  b'\x91`9:N\xd5@\xee\xa0\xad\xe4d6e\x92M'
IV:  b'L\xfe\x8dJ\x90\xc2nm\x1dk:\xa55u\xd1%'
IV:  b'\x98\x022\x84\xd8\xf9\x83\x18p\n\xf3\x9e\xb6\x01 \xaa'
IV:  b'\xfd\x02\xec\xaa\xa4\x91\xc9\xe3\x8a\xbb\xbf}D\xa7\xf02'
IV:  b'\xa1#\xd5.\xe5\xd80fA\x15\x0e\x9a\x83/\xde\x99'
IV:  b'\xb4\xc5\xb2\x03\n\x15\x80\x1d\xebG\x1a\x91\x96\x1c\xc4\xdc'
IV:  b'\x9eZ\x15;\x01\x92fG\x15\xdda\xb6x\xb8\x9eL'
IV:  b'\xe0\x8d\xe2\xdcg9@\xc8L\x01=\x7f9\x82\x14\x9f'
IV:  b"@'\xf8\xb9\x05\xbf\xd6\xf65\xf4\xac\xd2<\x08\xfd\xe1"