import logging
from flask import Flask, request, jsonify, render_template, session
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
model_path = r"C:\Users\Lenovo\Downloads\Model"
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

# session_data = {}

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
    match = re.search(r'(last|latest|top) (\d+) transactions', query.lower())
    if match:
        return int(match.group(2))
    return None


def extract_entities(text):
    doc = nlp(text)
    entities = {}

    # Convert GPE to PERSON and add to entities dictionary
    for ent in doc.ents:
        if ent.label_ == 'GPE' or ent.label_ == 'ORG':
            entities['PERSON'] = ent.text
        else:
            entities[ent.label_] = ent.text

    # Use regex to capture names if not recognized by SpaCy
    if "PERSON" not in entities:
        # Look for a pattern like "to [Name]"
        match = re.search(r'\bto\s+([A-Z][a-z]+)\b', text)
        if match:
            entities["PERSON"] = match.group(1)
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
    if 'all past transactions' in query.lower() or 'all transactions' in query.lower() or 'past transaction history' in query.lower():
        # Create a connection to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Define the query to fetch all transactions by name
        query = '''
        SELECT transaction_date, amount 
        FROM transactions
        WHERE sender_surname = ?
        ORDER BY transaction_date DESC
        '''

        # Execute the query with the name parameter
        cursor.execute(query, (sender_surname,))

        results = cursor.fetchall()
        conn.close()

        if results:
            transactions = [f"Date: {result[0]}, Amount: {result[1]:.2f}" for result in results]
            return f"All past transactions:\n" + "\n".join(transactions)
        else:
            return "No transactions found for the specified criteria."


    # Extract dates from the query
    duckling_entities = parse_with_duckling(query)
    if not duckling_entities:
        return "No date information found in the query."
    
    results = []
    now = datetime.now()
    start_date, end_date = None, None #new line added 

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

session_data = {}

def generate_response(user_input, name):
    intent = classify_intent(user_input)
    logging.debug(f"Detected intent: {intent}")

    entities = extract_entities(user_input)
    logging.debug(f"Extracted entities: {entities}")

    # Initialize chat history if not already present
    if name not in session_data:
        session_data[name] = {'chat_history': [], 'context': {}}

    # Append user query to chat history
    session_data[name]['chat_history'].append({'user': user_input, 'intent': intent, 'entities': entities})
    print(session_data)

    response = None  # Initialize response as None

    if intent == "check_balance":
        response = fetch_customer_balance(name)
    elif intent == "transfer_money":
        missing_entities = []
        if 'MONEY' not in entities and 'amount' not in session_data[name]['context']:
            missing_entities.append('MONEY')
        if 'PERSON' not in entities and 'receiver_name' not in session_data[name]['context']:
            missing_entities.append('PERSON')

        print('missing_entities',missing_entities)

        if not missing_entities:
            amount = float(entities['MONEY'].replace('$', '').replace(',', ''))
            receiver_name = entities['PERSON']
            # Clear the session context for this conversation
            session_data[name]['context'] = {}
            response = transfer_money(name, receiver_name, amount)
        else:
            if 'MONEY' not in entities and 'amount' not in session_data[name]['context']:
                session_data[name]['context']['missing'] = 'MONEY'
                if 'PERSON' in entities:
                    session_data[name]['context']['receiver_name'] = entities['PERSON']
                response = "How much money do you want to transfer?"
            elif 'PERSON' not in entities and 'receiver_name' not in session_data[name]['context']:
                session_data[name]['context']['missing'] = 'PERSON'
                response = "To whom do you want to transfer money?"
                if 'MONEY' in entities:
                    session_data[name]['context']['amount'] = float(entities['MONEY'].replace('$', '').replace(',', ''))
                response = "To whom do you want to transfer money?"


    elif intent == "inform" or intent == "affirm":
        missing_entity = session_data[name]['context'].get('missing')
        
        if missing_entity == 'MONEY' and 'MONEY' in entities:
            session_data[name]['context']['missing'] = None
            session_data[name]['context']['amount'] = float(entities['MONEY'].replace('$', '').replace(',', ''))
            if 'receiver_name' in session_data[name]['context']:
                receiver_name = session_data[name]['context'].pop('receiver_name')
                amount = session_data[name]['context'].pop('amount')
                response = transfer_money(name, receiver_name, amount)
            else:
                response = "To whom do you want to transfer money?"
                
        elif missing_entity == 'PERSON' and 'PERSON' in entities:
            session_data[name]['context']['missing'] = None
            session_data[name]['context']['receiver_name'] = entities['PERSON']
            if 'amount' in session_data[name]['context']:
                receiver_name = session_data[name]['context'].pop('receiver_name')
                amount = session_data[name]['context'].pop('amount')
                response = transfer_money(name, receiver_name, amount)
            else:
                response = "How much money do you want to transfer?"
                
        elif 'PERSON' in entities and 'MONEY' in entities:
            amount = float(entities['MONEY'].replace('$', '').replace(',', ''))
            receiver_name = entities['PERSON']
            response = transfer_money(name, receiver_name, amount)
        
        elif 'PERSON' in entities and 'amount' in session_data[name]['context']:
            receiver_name = entities['PERSON']
            amount = session_data[name]['context'].pop('amount')
            response = transfer_money(name, receiver_name, amount)
        
        elif 'MONEY' in entities and 'receiver_name' in session_data[name]['context']:
            amount = float(entities['MONEY'].replace('$', '').replace(',', ''))
            receiver_name = session_data[name]['context'].pop('receiver_name')
            response = transfer_money(name, receiver_name, amount)
        
        else:
            response = "Sorry, I don't understand your query."


    elif intent == "search_transactions":
        sender_surname = session_data.get("PERSON", name)
        if any(keyword in user_input.lower() for keyword in ["last", "latest", "top"]) and "transactions" in user_input.lower():
            response = fetch_transactions_by_name_and_date_expr(sender_surname, user_input)
        elif "all past transactions" in user_input.lower() or "all transactions" in user_input.lower() or "past transaction history" in user_input.lower():
            response = fetch_transactions_by_name_and_date_expr(sender_surname, user_input)
        else:
            date_expr = user_input
            response = fetch_transactions_by_name_and_date_expr(sender_surname, date_expr)
    elif intent == "check_human":
        response = "I am an AI created to assist you with financial inquiries."
    elif intent == 'open_account':
        response = "To open a new account, please visit our website or nearest branch with your identification documents."
    elif intent == 'close_account':
        response = "To close your account, please visit our nearest branch or contact our customer support."
    elif intent == "loan_inquiry":
        response = "You can inquire about loans and their eligibility criteria on our website or by visiting our branch."
    elif intent == 'credit_card_application':
        response = "You can apply for a credit card through our website or by visiting our nearest branch."
    elif intent == 'contact_support':
        response = "You can contact our customer support through the chat feature on our website or by calling our support number."

    # Check if response was set, otherwise return default response
    if response is None:
        response = "I'm sorry, I didn't understand your request. Can you please rephrase?"
        logging.debug("Default response used.")

    logging.debug(f"Final response: {response}")
    return response

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




