#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
from models.Nets import MLP, MNISTCNN as CNNMnist, CIFARCNN as CNNCifar
from models.Fed import FedAvg
from models.test import test_img
from kyber_py.ml_kem import ML_KEM_512
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
import pickle
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTabWidget, QProgressBar, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class FederatedLearningThread(QThread):
    log_signal = pyqtSignal(str, int)
    update_plot_signal = pyqtSignal(list)
    final_results_signal = pyqtSignal(float, float)

    def run(self):
        args = args_parser()
        args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

        # Load dataset and split users
        if args.dataset == 'mnist':
            trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
            dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
            dict_users = mnist_iid(dataset_train, args.num_users) if args.iid else mnist_noniid(dataset_train, args.num_users)
        elif args.dataset == 'cifar':
            trans_cifar = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
            dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
            dict_users = cifar_iid(dataset_train, args.num_users) if args.iid else exit('Error: only consider IID setting in CIFAR10')
        else:
            exit('Error: unrecognized dataset')
        
        img_size = dataset_train[0][0].shape

        # Build model
        if args.model == 'cnn' and args.dataset == 'cifar':
            net_glob = CNNCifar(args=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(args=args).to(args.device)
        elif args.model == 'mlp':
            len_in = np.prod(img_size)
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        net_glob.train()
        w_glob = net_glob.state_dict()
        loss_train = []

        # Generating key pair
        ek, dk = ML_KEM_512.keygen()    
        shared_key_sender, ct = ML_KEM_512.encaps(ek)
        shared_key_receiver = ML_KEM_512.decaps(dk, ct)
        assert shared_key_sender == shared_key_receiver

        aes_key = shared_key_receiver[:32]  
        iv = get_random_bytes(16)  
        init_cipher = AES.new(shared_key_sender[:32], AES.MODE_OFB, iv)
        another_cipher = AES.new(shared_key_receiver[:32], AES.MODE_OFB, iv)

        for iter in range(args.epochs):
            loss_locals = []
            w_locals = [w_glob for i in range(args.num_users)] if args.all_clients else []
            idxs_users = np.random.choice(range(args.num_users), max(int(args.frac * args.num_users), 1), replace=False)
            
            self.log_signal.emit("Training...", iter + 1) # Set Labels to "Waiting..."
            for idx in idxs_users:
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                
                w_serialized = pickle.dumps(w)
                ciphertext = init_cipher.encrypt(w_serialized)
                
                w_locals.append(copy.deepcopy(ciphertext))
                loss_locals.append(copy.deepcopy(loss))
            
            w_locals = [pickle.loads(another_cipher.decrypt(cipher)) for cipher in w_locals]
            w_glob = FedAvg(w_locals)
            net_glob.load_state_dict(w_glob)
            loss_avg = sum(loss_locals) / len(loss_locals)
            # Update the current epoch label with it's results
            self.log_signal.emit(f'Round {iter+1:3d}, Average loss {loss_avg:.3f}', iter + 1)
            loss_train.append(loss_avg)

        self.update_plot_signal.emit(loss_train) # update the plot

        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        self.final_results_signal.emit(acc_train, acc_test) # update the final results label

class FederatedLearningApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.start_training()

    def initUI(self):
        self.setWindowTitle("Federated Learning GUI")
        self.setGeometry(100, 100, 800, 600)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        layout = QVBoxLayout(self.central_widget)
        self.tabs = QTabWidget()

        self.output_tab = QWidget()
        self.output_layout = QVBoxLayout(self.output_tab)
        self.label_progress_bars = []
        for _ in range(10):
            hbox = QHBoxLayout()
            label = QLabel("Training...")
            progress_bar = QProgressBar()
            hbox.addWidget(label)
            hbox.addWidget(progress_bar)
            self.label_progress_bars.append((label, progress_bar))
            self.output_layout.addLayout(hbox)
        self.tabs.addTab(self.output_tab, "Training Results")

        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_tab)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)
        self.tabs.addTab(self.plot_tab, "Plot")

        layout.addWidget(self.tabs)

        self.final_acc_train = QLabel("Training accuracy: waiting...")
        self.final_acc_test = QLabel("Testing accuracy: waiting...")
        layout.addWidget(self.final_acc_train)
        layout.addWidget(self.final_acc_test)

    def start_training(self):
        self.thread = FederatedLearningThread()
        self.thread.log_signal.connect(self.log_message)
        self.thread.update_plot_signal.connect(self.update_plot)
        self.thread.final_results_signal.connect(self.update_final_results)

        self.progress_timers = []
        for label, progress_bar in self.label_progress_bars:
            progress_bar.setRange(0, 0)
            timer = QTimer(self)
            timer.timeout.connect(lambda bar=progress_bar: self.update_progress(bar))
            self.progress_timers.append(timer)
            timer.start(100)

        self.thread.start()

    def log_message(self, message, epoch):
        if 0 <= epoch - 1 < len(self.label_progress_bars):
            label, progress_bar = self.label_progress_bars[epoch - 1]
            label.setText(message)
            if message.startswith('Round'):
                progress_bar.setRange(0, 100)
                progress_bar.setValue(100)
            self.progress_timers[epoch - 1].stop()

    def update_progress(self, progress_bar):
        value = progress_bar.value() + 1
        if value > 99:
            value = 99
        progress_bar.setValue(value)

    def update_plot(self, loss_train):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(range(len(loss_train)), loss_train)
        ax.set_title('Training Loss Curve')
        ax.set_xlabel('Round')
        ax.set_ylabel('Train Loss')
        self.canvas.draw()

    def update_final_results(self, acc_train, acc_test):
        self.final_acc_train.setText(f"Training accuracy: {acc_train:.2f}")
        self.final_acc_test.setText(f"Testing accuracy: {acc_test:.2f}")

if __name__ == "__main__":
    app = QApplication([])
    window = FederatedLearningApp()
    window.show()
    app.exec_()
