import string

def remove_punctuation(text):
    # string.punctuation에는 모든 문장부호가 포함되어 있습니다.
    return text.translate(str.maketrans('', '', string.punctuation))
