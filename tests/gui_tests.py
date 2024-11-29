import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QPushButton

class SimpleApp(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the UI elements
        self.initUI()

    def initUI(self):
        # Create a vertical layout
        layout = QVBoxLayout(self)

        # Create the QTextEdit widget (where messages will appear)
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setPlaceholderText("Messages will appear here...")

        # Add the QTextEdit widget to the layout
        layout.addWidget(self.text_edit)

        # Create buttons to simulate PASS, FAIL, encryption, and decryption messages
        self.pass_button = QPushButton("Simulate PASS", self)
        self.fail_button = QPushButton("Simulate FAIL", self)
        self.encrypt_button = QPushButton("Simulate Encryption", self)
        self.decrypt_button = QPushButton("Simulate Decryption", self)

        # Connect buttons to methods that append messages
        self.pass_button.clicked.connect(self.show_pass_message)
        self.fail_button.clicked.connect(self.show_fail_message)
        self.encrypt_button.clicked.connect(self.show_encryption_message)
        self.decrypt_button.clicked.connect(self.show_decryption_message)

        # Add buttons to layout
        layout.addWidget(self.pass_button)
        layout.addWidget(self.fail_button)
        layout.addWidget(self.encrypt_button)
        layout.addWidget(self.decrypt_button)

        # Set the window title and geometry
        self.setWindowTitle('Simple PyQt5 Application')
        self.setGeometry(100, 100, 400, 400)

    def show_pass_message(self):
        """Simulate a PASS message and append to QTextEdit"""
        message = "User 1 integrity: PASS"
        self.text_edit.append(f"<b style='color: green'>{message}</b>")

    def show_fail_message(self):
        """Simulate a FAIL message and append to QTextEdit"""
        message = "User 1 integrity: FAIL"
        self.text_edit.append(f"<b style='color: red'>{message}</b>")

    def show_encryption_message(self):
        """Simulate an encryption message and append to QTextEdit"""
        message = "User 1: Encryption completed successfully."
        self.text_edit.append(f"<b style='color: blue'>{message}</b>")

    def show_decryption_message(self):
        """Simulate a decryption message and append to QTextEdit"""
        message = "User 1: Decryption completed successfully."
        self.text_edit.append(f"<b style='color: purple'>{message}</b>")

if __name__ == '__main__':
    # Create the application instance
    app = QApplication(sys.argv)

    # Create and display the window
    window = SimpleApp()
    window.show()

    # Run the application event loop
    sys.exit(app.exec_())