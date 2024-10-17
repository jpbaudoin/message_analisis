import chardet
import emoji
import pandas as pd
# detect the laguage of the text in the field mesaage_text
from langdetect import LangDetectException, detect, detect_langs
from unidecode import unidecode


def print_top(df, field, title, top=10):
    print(f"Top {top} {title} ({field}):")
    # get the top 10 most frequent values
    top_list = df[field].value_counts(dropna=False)
    top_list.index = top_list.index.fillna('N/A')
    top_list = top_list.sort_values(ascending=False)
    # count the total number of messages
    total = top_list.sum()
    #count the total without nan
    total_no_nan = top_list.sum() - top_list.get('N/A', 0)
    percentage = round((top_list / total) * 100, 2)
    percentage_no_nan = round((top_list / total_no_nan) * 100, 2)
    acum_percent = 0
    acum_percent_wna = 0
    order = 0
    for idx, count, percent in zip(top_list.index[0:top], top_list.values[0:top], percentage.values[0:top]):
        # add acumulated percentage
        acum_percent += percent
        order += 1 

        if total_no_nan == total:
            # print with two decimal places
            print(f"{order:02d} - {idx}: {count} \t ({percent}% - {acum_percent:.2f}%)")
        else:
            if idx == 'N/A':
                print(f"{order:02d} - {idx}: {count} \t ({percent}%  - {acum_percent:.2f}%) \t (NaN)")
            else:
                percent_wna = percentage_no_nan[idx]
                acum_percent_wna += percent_wna
                print(f"{order:02d} - {idx}: {count} \t ({percent}%  - {acum_percent:.2f}%) \t (W/NaN {percent_wna}% - {acum_percent_wna:.2f}%)")


def clean_lbr(text):
    # Replace continuous line breaks with a single line break
    cleaned_text = '\n'.join(line.strip() for line in text.splitlines() if line.strip())
    return cleaned_text

def remove_emojis(text):
    return emoji.replace_emoji(text, '')

def count_line_breaks(text):
    return text.count('\n')


def detect_language(text):
    try:
        return detect(text)
    except LangDetectException as e:
        print(e, text) 
        return "Unknown"
    else:
        print(e, text)
        return "Error langdetect"

def detect_multiple_languages(text):
    try:
        return detect_langs(text)
    except LangDetectException as e:
        print(e, text) 
        return "Unknown"
    else:
        print(e, text)
        return "Error langdetect"


def translate_chars(text, encoding, ext_chars_map=None, unidecode_on_miss=False):
    changed = False
    new_text_chars = []
    enc = chardet.detect(text.encode())['encoding']
    
    encoding_map = {
        "Windows-1254": "cp1254",
        "ISO-8859-9": "cp1254",
        "Windows-1252": "cp1252",
        "ISO-8859-1": "cp1252",
    }
    encoding = encoding_map.get(encoding, encoding)

    miss_text = text
    if unidecode_on_miss:
        miss_text = unidecode(text)
        # encode the text to utf-8
        miss_text = miss_text.encode().decode('utf-8')
        

    # Probably remove this and use the original encoding
    if encoding not in ext_chars_map:
        return miss_text
    
    char_map = ext_chars_map[encoding]

    
    if text in char_map:
        changed = True
        operation = char_map[text]['operation']

        if  operation == 'convert':
            new_text = char_map[text]['dst']
        elif operation == 'unicode':
            new_text = unidecode(text)
        else:
            # convert to utf-8 code
            utf_code = char_map[text]['dst']
            # char_code = "\xe2\x82\xac"
            # print(utf_code, type(utf_code))
            # print(char_code, type(char_code))

            # utf_code2 =to_raw_string(utf_code)
            # print(utf_code2)
            # char_bytes = bytes(utf_code2, 'latin1')
            char_bytes = utf_code.encode('latin1')

            print(char_bytes)
            new_text = char_bytes.decode('utf-8')
    else:
        return miss_text

    if changed:
        print("--------------------")
        print(new_text)
        print(text)
        print("--------------------")
    return new_text

def simple_utf_encode(text, encoding):
    try:
        text_encoded = text.encode(encoding)
    except UnicodeError as e:
        return None

    try:
        text_decoded = text_encoded.decode('utf-8')
    except UnicodeError as e:
        return None

    return text_decoded

def encode_utf8(text, encoding, gen_missed_chars=False, ext_chars_map=None):

    if pd.isna(text):
        return text

    
    try:
        text_encoded = text.encode(encoding)
    except UnicodeError as e:
        # split the text in two parts
        text1 = text[:e.start]
        text2 = text[e.end:]
        # encode the two parts
        text1_utf = encode_utf8(text1, encoding, gen_missed_chars, ext_chars_map)
        text2_utf = encode_utf8(text2, encoding, gen_missed_chars, ext_chars_map)
        # process missed characters
        missed_chars = text[e.start:e.end]
        missed_chars_utf = translate_chars(missed_chars, encoding, ext_chars_map, unidecode_on_miss=True)

        # appenf text to file
        if gen_missed_chars:
            with open("1_error_messages_encode.txt", "a") as f:
                f.write(f"{text[e.start:e.end]},{encoding}\n")

        return f"{text1_utf}{missed_chars_utf}{text2_utf}"

        # print(f"Error: {e}")
        # print(f"Error Start: {e.start}")
        # print(f"Error End: {e.end}")
        # print(f"Message: {text}")
        # print(f"Error: {text[e.start:e.end]}" )
        # print(f"BaseEncoding: {row['message_encoding']}")
        # print(f"Encoding: {encoding}")
        

        # print(f"Encoded2: {row['message_text_clean_utf8']}")
        # print("--------")

    try:
        text_decoded = text_encoded.decode('utf-8')
    except UnicodeError as e:

        text1 = text[:e.start]
        text2 = text[e.end:]
        # encode the two parts
        text1_utf = encode_utf8(text1, encoding, gen_missed_chars, ext_chars_map)
        text2_utf = encode_utf8(text2, encoding, gen_missed_chars, ext_chars_map)
        # process missed characters
        missed_chars = text[e.start:e.end]
        missed_chars_utf = translate_chars(missed_chars, encoding, ext_chars_map, unidecode_on_miss=True)

        # appenf text to file
        if gen_missed_chars:
            with open("2_error_messages_decode", "a") as f:
                f.write(f"{text[e.start:e.end]},{encoding}\n")

        return f"{text1_utf}{missed_chars_utf}{text2_utf}"

    return text_decoded