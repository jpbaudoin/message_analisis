import emoji
# detect the laguage of the text in the field mesaage_text
from langdetect import LangDetectException, detect, detect_langs


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

