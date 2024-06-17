import emoji
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

def remove_emojis(text):
    return emoji.replace_emoji(text, '')

def count_line_breaks(text):
    return text.count('\n')

file_path = '../dataset_diplodatos_completo.csv'
df = pd.read_csv(file_path)
df['source_posted_at'] = pd.to_datetime(df['source_posted_at'])

# Eliminar emojis y añadir columna message_no_emoji
df['message_no_emoji'] = df['message_text'].apply(remove_emojis).str.strip()

# Añadir columna con la longitud del mensaje sin emojis
df['message_length'] = df['message_no_emoji'].str.len()

# Añadir columna con el número de saltos de línea
df['message_lbr'] = df['message_no_emoji'].apply(count_line_breaks)

# Inicializar columna conversation con NaN
df['conversation'] = pd.NA

@app.route('/')
def index():
    chat_ids = df['chat_id'].unique()
    return render_template('index.html', chat_ids=chat_ids)

@app.route('/chat/<chat_id>')
def chat(chat_id):
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    telegram_user_id = request.args.get('telegram_user_id')
    conversation = request.args.get('conversation')
    min_length = request.args.get('min_length')
    max_length = request.args.get('max_length')
    min_lbr = request.args.get('min_lbr')
    max_lbr = request.args.get('max_lbr')
    commodity = request.args.get('commodity')
    
    chat_df = df[df['chat_id'] == float(chat_id)]
    
    if start_time:
        chat_df = chat_df[chat_df['source_posted_at'] >= pd.to_datetime(start_time)]
    if end_time:
        chat_df = chat_df[chat_df['source_posted_at'] <= pd.to_datetime(end_time)]
    if telegram_user_id:
        chat_df = chat_df[chat_df['telegram_user_id'] == float(telegram_user_id)]
    if conversation:
        chat_df = chat_df[chat_df['conversation'] == conversation]
    if min_length:
        chat_df = chat_df[chat_df['message_length'] >= int(min_length)]
    if max_length:
        chat_df = chat_df[chat_df['message_length'] <= int(max_length)]
    if min_lbr:
        chat_df = chat_df[chat_df['message_lbr'] >= int(min_lbr)]
    if max_lbr:
        chat_df = chat_df[chat_df['message_lbr'] <= int(max_lbr)]
    if commodity:
        chat_df = chat_df[chat_df['commodity'] == commodity]
    
    chat_df = chat_df.sort_values(by='source_posted_at')
    messages = chat_df.to_dict(orient='records')
    
    return render_template('chat.html', chat_id=chat_id, messages=messages, conversations=chat_df['conversation'].unique(), commodities=df['commodity'].unique())

if __name__ == '__main__':
    app.run(debug=True)
