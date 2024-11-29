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
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QLabel, QTabWidget, QProgressBar, QHBoxLayout, QTextEdit, QPushButton, QMessageBox
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Import hashlib for integrity validation
import hashlib

class FederatedLearningThread(QThread):
    log_signal = pyqtSignal(str, int)
    update_plot_signal = pyqtSignal(list)
    final_results_signal = pyqtSignal(float, float)
    process_status_signal = pyqtSignal(str)  # Signal for process status updates
    hash_check_signal = pyqtSignal(str)  # Add this line

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
            net_glob = CNNCifar(params=args).to(args.device)
        elif args.model == 'cnn' and args.dataset == 'mnist':
            net_glob = CNNMnist(params=args).to(args.device)
        elif args.model == 'mlp':
            len_in = np.prod(img_size)
            net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
        else:
            exit('Error: unrecognized model')

        net_glob.train()
        w_glob = net_glob.state_dict()
        loss_train = []
        self.process_status_signal.emit("Key pair generated successfully.")
        self.process_status_signal.emit("Encryption and decryption setup completed.")

        # Hex digests before and after transmission
        pre_transmission_vector = []
        post_transmission_vector = []

        for iter in range(args.epochs):
            loss_locals = []
            session_key_and_iv = []
            w_locals = [w_glob for i in range(args.num_users)] if args.all_clients else []
            idxs_users = np.random.choice(range(args.num_users), max(int(args.frac * args.num_users), 1), replace=False)

            self.log_signal.emit("Training...", iter + 1)  # Set Labels to "Waiting..."
            for idx in idxs_users:
                
                # Generating session keypair
                ek, dk = ML_KEM_512.keygen()
                # Generating shared secret key
                shared_key_sender, ct = ML_KEM_512.encaps(ek)
                # Key decapsulation
                shared_key_receiver = ML_KEM_512.decaps(dk, ct)

                iv = get_random_bytes(16)  # Generate random IV

                # Derive cipher for encryption from PEM
                init_cipher = AES.new(shared_key_sender[:32], AES.MODE_OFB, iv)
            
                # Append the receiver key and the iv to the dictionary
                session_key_and_iv.append((shared_key_receiver,iv,idx))
                
                local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
                w, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
                
                # Encryption
                self.process_status_signal.emit(f"Encryption: Model update for user {idx} encrypted successfully.")
                w_serialized = pickle.dumps(w)
                hash_value = hashlib.sha256(w_serialized).hexdigest() # pre-transmission hex digest
                pre_transmission_vector.append(hash_value)
                ciphertext = init_cipher.encrypt(w_serialized)

                if args.all_clients:
                    w_locals[idx] = copy.deepcopy(ciphertext)
                else:
                    w_locals.append(copy.deepcopy(ciphertext))
                
                loss_locals.append(copy.deepcopy(loss))
            
            # Decryption
            for i in range(len(w_locals)):
                shared_key_receiver, iv, idx = session_key_and_iv[i]
                
                self.process_status_signal.emit(f"Decryption: Model update for user {idx} decrypted successfully.")
                
                new_cipher = AES.new(shared_key_receiver[:32], AES.MODE_OFB, iv)
                decrypted = new_cipher.decrypt(w_locals[i])
                
                hash_value = hashlib.sha256(decrypted).hexdigest() # post-transmission hex digest
                post_transmission_vector.append(hash_value)
                
                w_locals[i] = pickle.loads(decrypted)
            
            w_glob = FedAvg(w_locals)
            
            net_glob.load_state_dict(w_glob)
            
            loss_avg = sum(loss_locals) / len(loss_locals)
            self.log_signal.emit(f'Round {iter+1:3d}, Average loss {loss_avg:.3f}', iter + 1)
            loss_train.append(loss_avg)
            
            # intrusion detection established by comparing sha256 hex digests before transmission and after transmission
            for i in range(len(w_locals)):
               #assert pre_transmission_vector[i] == post_transmission_vector[i]
                if pre_transmission_vector[i] == post_transmission_vector[i]:
                    self.hash_check_signal.emit(f"User {idx} integrity: PASS")
                else:
                    self.hash_check_signal.emit(f"User {idx} integrity: FAIL")
        
        self.update_plot_signal.emit(loss_train)  # update the plot

        acc_train, loss_train = test_img(net_glob, dataset_train, args)
        acc_test, loss_test = test_img(net_glob, dataset_test, args)
        self.final_results_signal.emit(acc_train, acc_test)  # update the final results label


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

        # Create a top-right horizontal layout for controls
        top_controls_layout = QHBoxLayout()
        top_controls_layout.addStretch()  # Push content to the right

        # Stop button
        self.stop_button = QPushButton("Stop")
        self.stop_button.setStyleSheet("background-color: red; color: white; font-weight: bold;")
        self.stop_button.clicked.connect(self.stop_training)  # Connect to stop_training method
        top_controls_layout.addWidget(self.stop_button)

        # Add top controls layout to the main layout
        layout.addLayout(top_controls_layout)

        # Output Tab
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

        # Plot Tab
        self.plot_tab = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_tab)
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)
        self.tabs.addTab(self.plot_tab, "Plot")

        # Process Status Tab
        self.status_tab = QWidget()
        self.status_layout = QHBoxLayout(self.status_tab)

        # Encryption status text browser
        self.encryption_status_text = QTextEdit()
        self.encryption_status_text.setReadOnly(True)
        self.encryption_status_text.setPlaceholderText("Encryption Messages")

        # Decryption status text browser
        self.decryption_status_text = QTextEdit()
        self.decryption_status_text.setReadOnly(True)
        self.decryption_status_text.setPlaceholderText("Decryption Messages")

        # Add both text browsers to the layout side-by-side
        self.status_layout.addWidget(self.encryption_status_text)
        self.status_layout.addWidget(self.decryption_status_text)

        # Add the process status tab to the tabs widget
        self.tabs.addTab(self.status_tab, "Process Status")

        layout.addWidget(self.tabs)

        # SIAS Tab
        self.sias_tab = QWidget()
        self.sias_layout = QVBoxLayout(self.sias_tab)
        self.sias_text = QTextEdit()
        self.sias_text.setReadOnly(True)
        self.sias_text.setPlaceholderText("Hash comparison results will appear here...")
        self.sias_layout.addWidget(self.sias_text)
        self.tabs.addTab(self.sias_tab, "SIAS")
        
        self.final_acc_train = QLabel("Training accuracy: waiting...")
        self.final_acc_test = QLabel("Testing accuracy: waiting...")
        layout.addWidget(self.final_acc_train)
        layout.addWidget(self.final_acc_test)

    def start_training(self):
        self.thread = FederatedLearningThread()
        self.thread.log_signal.connect(self.log_message)
        self.thread.update_plot_signal.connect(self.update_plot)
        self.thread.final_results_signal.connect(self.update_final_results)
        self.thread.process_status_signal.connect(self.update_process_status)  # Connect process status signal
        self.thread.hash_check_signal.connect(self.update_sias_log)  # Connect to SIAS tab

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
        current_value = progress_bar.value()
        progress_bar.setValue((current_value + 1) % 101)

    def update_plot(self, data):
        ax = self.figure.add_subplot(111)
        ax.clear()
        ax.plot(data, label="Training Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        self.canvas.draw()

    def update_final_results(self, acc_train, acc_test):
        self.final_acc_train.setText(f"Training accuracy: {acc_train:.2f}")
        self.final_acc_test.setText(f"Testing accuracy: {acc_test:.2f}")

    def update_process_status(self, message):
        if message.startswith("Encryption"):
            self.encryption_status_text.append(message)
        elif message.startswith("Decryption"):
            self.decryption_status_text.append(message)
            
    def update_sias_log(self, message):
        if "PASS" in message:
            styled_message = f'<span style="color: green; font-weight: bold;">{message}</span>'
        elif "FAIL" in message:
            styled_message = f'<span style="color: red; font-weight: bold;">{message}</span>'
        else:
            styled_message = message  # Default style if neither PASS nor FAIL is present
        self.sias_text.append(styled_message)

    def stop_training(self):
        reply = QMessageBox.question(
            self, "Confirm Stop", 
            "Are you sure you want to stop the training process?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            # Stop the thread safely
            if self.thread.isRunning():
                self.thread.terminate()
                self.thread.wait()
            QMessageBox.information(self, "Stopped", "The training process has been stopped.")
            self.close()  # Optionally close the application

if __name__ == "__main__":
    app = QApplication([])
    window = FederatedLearningApp()
    window.show()
    app.exec_()
