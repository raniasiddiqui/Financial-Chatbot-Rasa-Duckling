import logging
from flask import Flask, request, jsonify, render_template
from chatbot import classify_intent, extract_entities, fetch_customer_balance, transfer_money
import sqlite3
import torch
import pandas as pd
import spacy
from sklearn.preprocessing import LabelEncoder  # Ensure LabelEncoder is imported
from transformers import BertTokenizer, BertForSequenceClassification
from word2number import w2n
import re
from datetime import datetime, timedelta
import re
from dateutil.relativedelta import relativedelta
import sqlite3
from datetime import datetime
import requests
import json

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

# Load spaCy model
logging.info("Loading spaCy model...")
nlp = spacy.load("en_core_web_sm")

# Load BERT model and tokenizer
logging.info("Loading BERT model and tokenizer...")
model_path = r"C:\Users\Lenovo\Downloads\Model-20240704T051704Z-001\Model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load label encoder
logging.info("Loading label encoder...")
csv_file = r"C:\Users\Lenovo\Downloads\intents_and_exampless.csv"
df_intents = pd.read_csv(csv_file)
label_encoder = LabelEncoder()
df_intents['Intent'] = label_encoder.fit_transform(df_intents['Intent'])

session_data = {}

def classify_intent(query):
    logging.debug(f"Classifying intent for query: {query}")
    inputs = tokenizer.encode_plus(
        query,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, token_type_ids=None, attention_mask=attention_mask)

    logits = outputs.logits
    intent = torch.argmax(logits, dim=1).cpu().numpy()[0]
    classified_intent = label_encoder.inverse_transform([intent])[0]
    logging.debug(f"Classified intent: {classified_intent}")
    return classified_intent

def parse_with_duckling(text):
    url = "http://localhost:8000/parse"
    data = {
        "text": text,
        "locale": "en_US",
        "tz": "UTC"
    }
    headers = {
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8"
    }

    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        print(response.json())
        return response.json()
    else:
        logging.error(f"Duckling request failed with status code {response.status_code}")
        return None
    
def extract_number_from_query(query):
    match = re.search(r'last (\d+) transactions', query.lower())
    if match:
        return int(match.group(1))
    return None


def extract_entities(text):
    # logging.debug(f"Extracting entities from text: {text}")
    doc = nlp(text)
    entities = {ent.label_: ent.text for ent in doc.ents}

    # Use regex to capture names if not recognized by SpaCy
    if "PERSON" not in entities:
        # Look for patterns like "to [Name]", "Send [Name]", or "pay [Name]"
        match = re.search(r'\b(?:to|send|pay)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)\b', text, re.IGNORECASE)
        if match:
            entities["PERSON"] = match.group(1)
    
    # Print the extracted entities
    print("Extracted entities:", entities)
    
    return entities


def fetch_transactions_by_name_and_date_expr(sender_surname, query, db_path=r"C:\Users\Lenovo\Downloads\chatbot_tranc.db"):
    # Check if the query requests a specific number of latest transactions
    num_transactions = extract_number_from_query(query)
    if num_transactions:
        # Create a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Define the query to fetch the latest X transactions by name
        query = '''
        SELECT transaction_date, amount 
        FROM transactions
        WHERE sender_surname = ?
        ORDER BY transaction_date DESC
        LIMIT ?
        '''

        # Execute the query with the name and number of transactions parameters
        cursor.execute(query, (sender_surname, num_transactions))

        results = cursor.fetchall()
        conn.close()

        if results:
            transactions = [f"Date: {result[0]}, Amount: {result[1]:.2f}" for result in results]
            return f"Latest {num_transactions} transactions:\n" + "\n".join(transactions)
        else:
            return "No transactions found for the specified criteria."

    # Extract dates from the query
    duckling_entities = parse_with_duckling(query)
    if not duckling_entities:
        return "No date information found in the query."
    
    results = []
    now = datetime.now()

    # Handle common date expressions manually
    if 'last month' in query.lower():
        start_date = (now.replace(day=1) - timedelta(days=1)).replace(day=1)
        end_date = now.replace(day=1) - timedelta(days=1)
    elif 'this month' in query.lower():
        start_date = now.replace(day=1)
        end_date = now
    elif 'current month' in query.lower():
        start_date=now.replace(day=1)
        end_date = now
    elif 'last week' in query.lower():
        start_date = (now - timedelta(days=now.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = start_date + timedelta(days=6)
    elif 'this week' in query.lower():
        start_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = now
    elif 'current week' in query.lower():
        start_date = (now - timedelta(days=now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        end_date = now
    elif 'last year' in query.lower():
        start_date = (now.replace(month=1, day=1) - timedelta(days=1)).replace(month=1, day=1)
        end_date = now.replace(month=1, day=1) - timedelta(days=1)
    elif 'this year' in query.lower():
        start_date = now.replace(month=1, day=1)
        end_date = now
    elif 'current year' in query.lower():
        start_date = now.replace(month=1, day=1)
        end_date = now
    elif 'all past transactions' in query.lower() or 'all transactions' in query.lower() or 'past transaction history' in query.lower():
        start_date = datetime.min
        end_date = now
    else:
        for entity in duckling_entities:
            if entity['dim'] == 'time':
                value = entity['value']
                if value['type'] == 'interval':
                    start_date = datetime.fromisoformat(value['from']['value'].split('T')[0])
                    end_date = datetime.fromisoformat(value['to']['value'].split('T')[0]) if 'to' in value else (start_date.replace(month=start_date.month % 12 + 1, day=1) - timedelta(days=1)) if start_date.month != 12 else start_date.replace(month=12, day=31)
                else:
                    # Handle the case where only a month or a specific date is given
                    date = datetime.fromisoformat(value['value'].split('T')[0])
                    start_date = date.replace(day=1)
                    end_date = (date.replace(month=date.month % 12 + 1, day=1) - timedelta(days=1)) if date.month != 12 else date.replace(month=12, day=31)
                break

    if not start_date or not end_date:
        return "Unable to determine date range from the provided query."

    # Convert the target dates to string format
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Create a connection to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Define the query to fetch transactions by name and date range
    query = '''
    SELECT COUNT(*) cnt, SUM(amount) total_amount 
    FROM transactions
    WHERE sender_surname = ? AND transaction_date BETWEEN ? AND ?
    '''

    # Execute the query with the name and date parameters
    cursor.execute(query, (sender_surname, start_date_str, end_date_str))

    result = cursor.fetchone()
    conn.close()
    
    if result:
        count, total_amount = result
        return f"Total number of transactions: {count}, Sum of all transactions: {total_amount:.2f}"
    else:
        return "No transactions found for the specified criteria."
 


def fetch_customer_balance(surname):
    logging.debug(f"Fetching customer balance for surname: {surname}")
    conn = sqlite3.connect(r"C:\Users\Lenovo\Downloads\chatbot_tranc.db")
    c = conn.cursor()
    c.execute('SELECT Balance FROM customers WHERE Surname = ?', (surname,))
    result = c.fetchone()
    conn.close()
    if result:
        balance_message = f"Your current balance is ${result[0]}."
    else:
        balance_message = "No balance data available."
    logging.debug(balance_message)
    return balance_message

def transfer_money(name, receiver_name, amount):
    logging.debug(f"Transferring money from {name} to {receiver_name} amount: {amount}")
    conn = sqlite3.connect(r"C:\Users\Lenovo\Downloads\chatbot_tranc.db")
    c = conn.cursor()

    # Fetch sender's balance by surname
    c.execute('SELECT Balance FROM customers WHERE Surname = ?', (name,))
    sender_balance = c.fetchone()

    # Fetch receiver's balance by surname
    c.execute('SELECT Balance FROM customers WHERE Surname = ?', (receiver_name,))
    receiver_balance = c.fetchone()

    if not receiver_balance:
        conn.close()
        return f"Receiver with Name {receiver_name} not found."
    if not sender_balance or sender_balance[0] < amount:
        conn.close()
        return f"Insufficient funds."

    try:
        # Begin transaction
        logging.debug(f"Sender's balance before transfer: ${sender_balance[0]}")
        logging.debug(f"Receiver's balance before transfer: ${receiver_balance[0]}")
        conn.execute('BEGIN TRANSACTION')

        # Decrease balance of sender
        c.execute('UPDATE customers SET Balance = Balance - ? WHERE Surname = ?', (amount, name))

        # Increase balance of receiver
        c.execute('UPDATE customers SET Balance = Balance + ? WHERE Surname = ?', (amount, receiver_name))

        # Commit transaction
        conn.commit()

        c.execute('SELECT Balance FROM customers WHERE Surname = ?', (name,))
        updated_sender_balance = c.fetchone()

        c.execute('SELECT Balance FROM customers WHERE Surname = ?', (receiver_name,))
        updated_receiver_balance = c.fetchone()

        # Print balances after the transaction
        logging.debug(f"Sender's balance after transfer: ${updated_sender_balance[0]}")
        logging.debug(f"Receiver's balance after transfer: ${updated_receiver_balance[0]}")

        conn.close()
        return f"Successfully transferred ${amount} to {receiver_name}."
    except sqlite3.Error as e:
        # Rollback transaction in case of error
        conn.rollback()
        conn.close()
        return f"Failed to transfer money due to: {str(e)}"


def generate_response(user_input, name):
    global session_data
    logging.debug(f"Generating response for user input: {user_input}")
    intent = classify_intent(user_input)
    entities = extract_entities(user_input)

    # Update session data with new entities
    session_data.update(entities)

    if intent == "check_balance":
        return fetch_customer_balance(name)
    elif intent == "transfer_money":
        amount = session_data.get("MONEY", None)
        receiver_name = session_data.get("PERSON", None)

        if amount and receiver_name:
            # Clear the session data after use
            session_data.clear()
            amount = float(amount.replace('$', '').replace(',', ''))
            return transfer_money(name, receiver_name, amount)
        else:
            return "Please provide receiver's name and amount to transfer."
    elif intent == "search_transactions":
        sender_surname = session_data.get("PERSON", name)
        # Check if the query requests the latest X number of transactions
        if "last" in user_input.lower() and "transactions" in user_input.lower():
            return fetch_transactions_by_name_and_date_expr(sender_surname, user_input)
        else:
            date_expr = user_input
            return fetch_transactions_by_name_and_date_expr(sender_surname, date_expr)
    elif intent == "check_human":
        return "I am an AI created to assist you with financial inquiries."
    elif intent == 'open_account':
        return "To open a new account, please visit our website or nearest branch with your identification documents."
    elif intent == 'close_account':
        return "To close your account, please visit our nearest branch or contact our customer support."
    elif intent == "loan_inquiry":
        return "You can inquire about loans and their eligibility criteria on our website or by visiting our branch."
    elif intent == 'credit_card_application':
        return "You can apply for a credit card through our website or by visiting our nearest branch."
    elif intent == 'contact_support':
        return "You can contact our customer support through the chat feature on our website or by calling our support number."
    else:
        return "Sorry, I don't understand your query."


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('user_input')
    name = data.get('name')
    logging.debug(f"Received user input: {user_input} from user: {name}")
    response = generate_response(user_input, name)
    logging.debug(f"Generated response: {response}")
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

