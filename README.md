## ✨ Financial Chatbot with Rasa Duckling

This project is a Flask-based financial chatbot that allows users to perform various banking tasks, such as checking account balances, transferring money, fetching reward points, searching transactions, and finding branch locations. The chatbot uses a BERT model for intent classification, spaCy for entity extraction, and Rasa Duckling for parsing date and time entities. It interacts with a SQLite database to store and retrieve user and transaction data.

## 💫 Features


- Intent Classification: Uses a pre-trained BERT model to classify user intents (e.g., check_balance, transfer_money, search_transactions).



- Entity Extraction: Leverages spaCy for named entity recognition and regex for extracting names and amounts.



- Date Parsing: Integrates with Rasa Duckling to parse date and time expressions (e.g., "last week", "this month").



- Database Integration: Connects to a SQLite database (chatbot_tranc.db) to manage customer data, transactions, and branch locations.



- Session Management: Maintains conversation context using a session_data dictionary for multi-turn interactions (e.g., money transfer confirmation and OTP verification).



- Web Interface: Provides a simple HTML interface (index.html) for user interaction via a web browser.

### Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7+: Required for running the Flask app and its dependencies.

- Docker: Needed to run the Rasa Duckling server for date parsing.

- pip: Python package manager to install dependencies.

- SQLite Database: The database file (chatbot_tranc.db) located at C:\Users\Lenovo\Downloads\chatbot_tranc.db.


- BERT Model: Pre-trained BERT model and tokenizer located at C:\Users\Lenovo\Downloads\MODEL_BERTT-\MODEL_BERTT.

- Intents CSV: Intent data file (_intents-and-examples.csv) located at C:\Users\Lenovo\Downloads\_intents-and-examples.csv.
## How to run this file 
- Download the intents and examples csv file and replace the path in line 39 of the app.py file.
- Download the model and replace the path in line 31 of the app.py file.
- Download the chatbot-transc database file and replace the path in app.py file.
Then Install Docker and run the following commands
- Open terminal and docker pull rasa/duckling:latest
- Open CMD and docker run -p 8000:8000 rasa/duckling:latest
- And then run python app.py! 
