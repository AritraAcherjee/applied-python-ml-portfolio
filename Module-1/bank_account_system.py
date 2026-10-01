import os


# Each BankAccount object stores account details and writes its activity to a text file
class BankAccount:
    def __init__(self, name, accountType, balance=0):
        self.name = name
        self.accountType = accountType
        self.balance = balance

        # Use the number of existing text files to assign the next account number
        txt_files = [file for file in os.listdir() if file.endswith(".txt")]
        self.accountNumber = len(txt_files) + 1

        # Include the account number, type, and name so each transaction file is easy to identify
        self.filename = (
            str(self.accountNumber)
            + "_"
            + self.accountType
            + "_"
            + self.name
            + ".txt"
        )

        # Create a fresh transaction-history file when the account is opened
        with open(self.filename, "w") as file:
            file.write("Account created\n")
            file.write("Starting balance: $" + str(self.balance) + "\n")


    def deposit(self, amount):
        # Deposits must be positive before the balance and transaction history are updated
        if amount > 0:
            self.balance += amount

            with open(self.filename, "a") as file:
                file.write("Deposit: $" + str(amount) + "\n")
        else:
            print("Deposit amount must be greater than zero.")


    def withdraw(self, amount):
        # Reject invalid withdrawals before checking whether enough money is available
        if amount <= 0:
            print("Withdrawal amount must be greater than zero.")

        # Prevent withdrawals that would reduce the account below zero
        elif amount > self.balance:
            print("Insufficient funds.")

        else:
            self.balance -= amount

            with open(self.filename, "a") as file:
                file.write("Withdrawal: $" + str(amount) + "\n")


    def get_balance(self):
        return self.balance


    def get_user_id(self):
        return self.accountNumber


    def get_username(self):
        return self.name


    def get_account_type(self):
        return self.accountType


    # Read the saved file so the complete account history can be displayed when requested
    def get_transaction_history(self):
        with open(self.filename, "r") as file:
            return file.read()


# Create sample accounts and perform transactions to demonstrate the class methods
account1 = BankAccount("John", "checking")
account2 = BankAccount("Sarah", "saving", 500)

account1.deposit(100)
account1.deposit(50)
account1.withdraw(25)

account2.deposit(200)
account2.withdraw(100)
account2.withdraw(1000)

print("Account 1")
print("Name:", account1.get_username())
print("User ID:", account1.get_user_id())
print("Account Type:", account1.get_account_type())
print("Balance:", account1.get_balance())
print("Transaction History:")
print(account1.get_transaction_history())



print("Account 2")
print("Name:", account2.get_username())
print("User ID:", account2.get_user_id())
print("Account Type:", account2.get_account_type())
print("Balance:", account2.get_balance())
print("Transaction History:")
print(account2.get_transaction_history())