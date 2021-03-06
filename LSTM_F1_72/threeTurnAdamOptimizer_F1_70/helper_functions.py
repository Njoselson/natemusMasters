import emoji
import re

def is_emoji(s):
    return s in emoji.UNICODE_EMOJI

def remove_emoji(text):
    letter_array = [char for char in text if not is_emoji(char)]
    return ''.join(letter_array)

def add_space(text):
    return ''.join(' ' + char + ' ' if is_emoji(char) else char for char in text).strip()

def remove_text(text):
    words = text.split(' ')
    emojis = [word for word in words if is_emoji(word)]
    return emojis

def count_length(text):
    return len(text)

def count_upper_case(text):
    return sum(1 for c in text if c.isupper())

def find_converage(big_dict, small_dict):
    not_in_small = {}
    missing = 0
    exist_both = 0
    for key in small_dict:
        if key in big_dict:
            exist_both += 1
        else:
            not_in_small[key] = 0
            missing += 1
    coverage = exist_both/(exist_both + missing)
    return [coverage, missing, exist_both], not_in_small

def add_word(string, word):
    return 1 if word in string.split(' ') else 0

def find_number_missing(not_in_small, tokenizer_input_data):
    loop_count = 0
    for line in tokenizer_input_data:
        for key, _ in not_in_small.items():
            not_in_small[key] += add_word(line, key)
        loop_count += 1
        print("%s of %s"%(loop_count, len(tokenizer_input_data)))
    return not_in_small

def fix_apos(word):
    replaced_word = re.sub(r"(it['|´|’]s)", "it is", word)
    replaced_word = re.sub(r"(can['|´|’]t)", "can not", replaced_word)
    replaced_word = re.sub(r"(ain['|´|’]t)", "am not", replaced_word)
    replaced_word = re.sub(r"([I|a-z]{0,10})['|´|’]re", r"\1 are", replaced_word)
    replaced_word = re.sub(r"([I|a-z]{0,10})['|´|’]ll", r"\1 will", replaced_word)
    replaced_word = re.sub(r"([I|a-z]{0,10})['|´|’]ve", r"\1 have", replaced_word)
    replaced_word = re.sub(r"(he|who|how|when|there)['|´|’]s", r"\1 is", replaced_word)
    replaced_word = re.sub(r"that['|´|’]s", r"that is", replaced_word)
    replaced_word = re.sub(r"what['|´|’]s", r"what is", replaced_word)
    replaced_word = re.sub(r"(let['|´|’]s)", "let us", replaced_word)
    replaced_word = re.sub(r"([I|i]['|´|’]m)", "i am", replaced_word)
    replaced_word = re.sub(r"(won['|´|’]t)", "will not", replaced_word)
    replaced_word = re.sub(r"(n['|´|’]t)", " not", replaced_word)
    replaced_word = re.sub(r"['|´|’]", r" ", replaced_word)
    return replaced_word

def str2emoji(text):
    text = re.sub(r":‑\)"," ☺️ ",text)
    text = re.sub(r":\)\)\)\)"," ☺️ ",text)
    text = re.sub(r":\)\)\)"," ☺️ ",text)
    text = re.sub(r":\)\)"," ☺️ ",text)
    text = re.sub(r":\)"," ☺️ ",text)
    text = re.sub(r":-\]"," ☺️ ",text)
    text = re.sub(r":\]"," ☺️ ",text)
    text = re.sub(r":-3"," ☺️ ",text)
    text = re.sub(r":3","  ☺️ ",text)
    text = re.sub(r":->"," ☺️ ",text)
    text = re.sub(r"</3",' 💔 ',text)
    text = re.sub(r"<3"," ❤️ ",text)
    text = re.sub(r":>"," ☺️ ",text)
    text = re.sub(r"8-\)"," ☺️ ",text)
    text = re.sub(r":o\)"," ☺️ ",text)
    text = re.sub(r":-\}"," ☺️ ",text)
    text = re.sub(r":\}"," ☺️ ",text)
    text = re.sub(r":-\)"," ☺️ ",text)
    text = re.sub(r":c\)"," ☺️ ",text)
    text = re.sub(r":\^\)"," ☺️ ",text)
    text = re.sub(r"=\]"," ☺️ ",text)
    text = re.sub(r"=\)"," ☺️ ",text)
    text = re.sub(r":‑D"," 😃 ",text)
    text = re.sub(r":D"," 😃 ",text)
    text = re.sub(r"8‑D"," 😃 ",text)
    text = re.sub(r"8D"," 😃 ",text)
    text = re.sub(r"X‑D"," 😃 ",text)
    text = re.sub(r"XD"," 😃 ",text)
    text = re.sub(r"=D"," 😃 ",text)
    text = re.sub(r"=3"," 😃 ",text)
    text = re.sub(r"B\^D"," 😃 ",text)
    text = re.sub(r":-\)\)"," 😃 ",text)
    text = re.sub(r":‑\("," ☹️ ",text)
    text = re.sub(r":-\("," ☹️ ",text)
    text = re.sub(r":-d", " 🤤 ",text)
    text = re.sub(r":d", " 🤤 ",text)
    text = re.sub(r"=d", " 🤤 ",text)

    text = re.sub(r":\("," ☹️ ",text)
    text = re.sub(r":‑c"," ☹️ ",text)
    text = re.sub(r":c"," ☹️ ",text)
    text = re.sub(r":‑<"," ☹️ ",text)
    text = re.sub(r":<"," ☹️ ",text)
    text = re.sub(r":‑\["," ☹️ ",text)
    text = re.sub(r":\["," ☹️ ",text)
    text = re.sub(r":-\|\|"," ☹️ ",text)
    text = re.sub(r">:\["," ☹️ ",text)
    text = re.sub(r":\{"," ☹️ ",text)
    text = re.sub(r":@"," ☹️ ",text)
    text = re.sub(r">:\("," ☹️ ",text)
    text = re.sub(r":'‑\("," 😭 ",text)
    text = re.sub(r":'\("," 😭 ",text)
    text = re.sub(r":'‑\)"," 😃 ",text)
    text = re.sub(r":'\)"," 😃 ",text)
    text = re.sub(r"D‑':"," 😧 ",text)
    text = re.sub(r"D:<"," 😨 ",text)
    text = re.sub(r"D:"," 😧 ",text)
    text = re.sub(r"D8"," 😧 ",text)
    text = re.sub(r"D;"," 😧 ",text)
    text = re.sub(r"D="," 😧 ",text)
    text = re.sub(r"DX"," 😧 ",text)
    text = re.sub(r":‑O"," 😮 ",text)
    text = re.sub(r":O"," 😮 ",text)
    text = re.sub(r":‑o"," 😮 ",text)
    text = re.sub(r":o"," 😮 ",text)
    text = re.sub(r":-0"," 😮 ",text)
    text = re.sub(r"8‑0"," 😮 ",text)
    text = re.sub(r">:O"," 😮 ",text)
    text = re.sub(r":-\*"," 😗 ",text)
    text = re.sub(r":\*"," 😗 ",text)
    text = re.sub(r":X"," 😗 ",text)
    text = re.sub(r";‑\)"," 😉 ",text)
    text = re.sub(r";\)"," 😉 ",text)
    text = re.sub(r"\*-\)"," 😉 ",text)
    text = re.sub(r"\*\)"," 😉 ",text)
    text = re.sub(r";‑\]"," 😉 ",text)
    text = re.sub(r";\]"," 😉 ",text)
    text = re.sub(r";\^\)"," 😉 ",text)
    text = re.sub(r":‑,"," 😉 ",text)
    text = re.sub(r";D"," 😉 ",text)
    text = re.sub(r":‑P"," 😛 ",text)
    text = re.sub(r":‑p"," 😛 ",text)
    text = re.sub(r":P"," 😛 ",text)
    text = re.sub(r":p"," 😛 ",text)
    text = re.sub(r"X‑P"," 😛 ",text)
    text = re.sub(r":‑Þ"," 😛 ",text)
    text = re.sub(r":Þ"," 😛 ",text)
    text = re.sub(r":b"," 😛 ",text)
    text = re.sub(r"d:"," 😛 ",text)
    text = re.sub(r"=p"," 😛 ",text)
    text = re.sub(r">:P"," 😛 ",text)
    text = re.sub(r":‑/"," 😕 ",text)
    text = re.sub(r":/"," 😕 ",text)
    text = re.sub(r":-\[\.\]"," 😕 ",text)
    text = re.sub(r">:/"," 😕 ",text)
    text = re.sub(r"=/"," 😕 ",text)
    text = re.sub(r":L"," 😕 ",text)
    text = re.sub(r"=L"," 😕 ",text)
    text = re.sub(r":S"," 😕 ",text)
    text = re.sub(r":‑\|"," 😐 ",text)
    text = re.sub(r":\|"," 😐 ",text)
    text = re.sub(r":$"," 😳 ",text)
    text = re.sub(r":‑x"," 🤐 ",text)
    text = re.sub(r":x"," 🤐 ",text)
    text = re.sub(r":‑#"," 🤐 ",text)
    text = re.sub(r":#"," 🤐 ",text)
    text = re.sub(r":‑&"," 🤐 ",text)
    text = re.sub(r":&"," 🤐 ",text)
    text = re.sub(r"O:‑\)"," 😇 ",text)
    text = re.sub(r"O:\)"," 😇 ",text)
    text = re.sub(r"0:‑3"," 😇 ",text)
    text = re.sub(r"0:3"," 😇 ",text)
    text = re.sub(r"0:‑\)"," 😇 ",text)
    text = re.sub(r"0:\)"," 😇 ",text)
    text = re.sub(r":‑b"," 😛 ",text)
    text = re.sub(r"0;\^\)"," 😇 ",text)
    text = re.sub(r">:‑\)"," 😈 ",text)
    text = re.sub(r">:\)"," 😈 ",text)
    text = re.sub(r"\}:‑\)"," 😈 ",text)
    text = re.sub(r"\}:\)"," 😈 ",text)
    text = re.sub(r"3:‑\)"," 😈 ",text)
    text = re.sub(r"3:\)"," 😈 ",text)
    text = re.sub(r">;\)"," 😈 ",text)
    text = re.sub(r"\|;‑\)"," 😎 ",text)
    text = re.sub(r"\|‑O"," 😏 ",text)
    text = re.sub(r":‑J"," 😏 ",text)
    text = re.sub(r"%‑\)"," 😵 ",text)
    text = re.sub(r"%\)"," 😵 ",text)
    text = re.sub(r":-###.."," 🤒 ",text)
    text = re.sub(r":###.."," 🤒 ",text)
    text = re.sub(r"\(>_<\)"," 😣 ",text)
    text = re.sub(r"\(>_<\)>"," 😣 ",text)
    text = re.sub(r"\(';'\)"," 👶 ",text)
    text = re.sub(r"\(\^\^>``"," 😓 ",text)
    text = re.sub(r"\(\^_\^;\)"," 😓 ",text)
    text = re.sub(r"\(-_-;\)"," 😓 ",text)
    text = re.sub(r"\(~_~;\) \(・\.・;\)"," 😓 ",text)
    text = re.sub(r"\(-_-\)zzz"," 😴 ",text)
    text = re.sub(r"\(\^_-\)"," 😉 ",text)
    text = re.sub(r"\(\(\+_\+\)\)"," 😕 ",text)
    text = re.sub(r"\(\+o\+\)"," 😕 ",text)
    text = re.sub(r"\^_\^"," 😃 ",text)
    text = re.sub(r"\(\^_\^\)/"," 😃 ",text)
    text = re.sub(r"\(\^O\^\)／"," 😃 ",text)
    text = re.sub(r"\(\^o\^\)／"," 😃 ",text)
    text = re.sub(r"\(__\)"," 🙇 ",text)
    text = re.sub(r"_\(\._\.\)_"," 🙇 ",text)
    text = re.sub(r"<\(_ _\)>"," 🙇 ",text)
    text = re.sub(r"<m\(__\)m>"," 🙇 ",text)
    text = re.sub(r"m\(__\)m"," 🙇 ",text)
    text = re.sub(r"m\(_ _\)m"," 🙇 ",text)
    text = re.sub(r"\('_'\)"," 😭 ",text)
    text = re.sub(r"\(/_;\)"," 😭 ",text)
    text = re.sub(r"\(T_T\) \(;_;\)"," 😭 ",text)
    text = re.sub(r"\(;_;"," 😭 ",text)
    text = re.sub(r"\(;_:\)"," 😭 ",text)
    text = re.sub(r"\(;O;\)"," 😭 ",text)
    text = re.sub(r"\(:_;\)"," 😭 ",text)
    text = re.sub(r"\(ToT\)"," 😭 ",text)
    text = re.sub(r";_;"," 😭 ",text)
    text = re.sub(r";-;"," 😭 ",text)
    text = re.sub(r";n;"," 😭 ",text)
    text = re.sub(r";;"," 😭 ",text)
    text = re.sub(r"Q\.Q"," 😭 ",text)
    text = re.sub(r"T\.T"," 😭 ",text)
    text = re.sub(r"QQ"," 😭 ",text)
    text = re.sub(r"Q_Q"," 😭 ",text)
    text = re.sub(r"\(-\.-\)"," 😞 ",text)
    text = re.sub(r"\(-_-\)"," 😞 ",text)
    text = re.sub(r"-_-"," 😞 ",text)
    text = re.sub(r"\(一一\)"," 😞 ",text)
    text = re.sub(r"\(；一_一\)"," 😞 ",text)
    text = re.sub(r"\(=_=\)"," 😩 ",text)
    text = re.sub(r"\(=\^\·\^=\)"," 😺 ",text)
    text = re.sub(r"\(=\^\·\·\^=\)"," 😺 ",text)
    text = re.sub(r"=_\^= "," 😺 ",text)
    text = re.sub(r"\(\.\.\)"," 😔 ",text)
    text = re.sub(r"\(\._\.\)"," 😔 ",text)
    text = re.sub(r"\(\・\・\?"," 😕 ",text)
    text = re.sub(r"\(\?_\?\)"," 😕 ",text)
    text = re.sub(r">\^_\^<"," 😃 ",text)
    text = re.sub(r"<\^!\^>"," 😃 ",text)
    text = re.sub(r"\^/\^"," 😃 ",text)
    text = re.sub(r"\（\*\^_\^\*）"," 😃 ",text)
    text = re.sub(r"\(\^<\^\) \(\^\.\^\)"," 😃 ",text)
    text = re.sub(r"\(^\^\)"," 😃 ",text)
    text = re.sub(r"\(\^\.\^\)"," 😃 ",text)
    text = re.sub(r"\(\^_\^\.\)"," 😃 ",text)
    text = re.sub(r"\(\^_\^\)"," 😃 ",text)
    text = re.sub(r"\(\^\^\)"," 😃 ",text)
    text = re.sub(r"\(\^J\^\)"," 😃 ",text)
    text = re.sub(r"\(\*\^\.\^\*\)"," 😃 ",text)
    text = re.sub(r"\(\^—\^\）"," 😃 ",text)
    text = re.sub(r"\(#\^\.\^#\)"," 😃 ",text)
    text = re.sub(r"\（\^—\^\）"," 👋 ",text)
    text = re.sub(r"\(;_;\)/~~~"," 👋 ",text)
    text = re.sub(r"\(\^\.\^\)/~~~"," 👋 ",text)
    text = re.sub(r"\(T_T\)/~~~"," 👋 ",text)
    text = re.sub(r"\(\*\^0\^\*\)"," 😍 ",text)
    text = re.sub(r"\(\*_\*\)"," 😍 ",text)
    text = re.sub(r"\(\*_\*;"," 😍 ",text)
    text = re.sub(r"\(\+_\+\) \(@_@\)"," 😍 ",text)
    text = re.sub(r"\(\*\^\^\)v"," 😂 ",text)
    text = re.sub(r"\(\^_\^\)v"," 😂 ",text)
    text = re.sub(r"\(ーー;\)"," 😓 ",text)
    text = re.sub(r"\(\^0_0\^\)"," 😎 ",text)
    text = re.sub(r"\(\＾ｖ\＾\)"," 😀 ",text)
    text = re.sub(r"\(\＾ｕ\＾\)"," 😀 ",text)
    text = re.sub(r"\(\^\)o\(\^\)"," 😀 ",text)
    text = re.sub(r"\(\^O\^\)"," 😀 ",text)
    text = re.sub(r"\(\^o\^\)"," 😀 ",text)
    text = re.sub(r"\)\^o\^\("," 😀 ",text)
    text = re.sub(r":O o_O"," 😮 ",text)
    text = re.sub(r"o_0"," 😮 ",text)
    text = re.sub(r"o\.O"," 😮 ",text)
    text = re.sub(r"\(o\.o\)"," 😮 ",text)
    text = re.sub(r"oO"," 😮 ",text)
    text = re.sub(r':\‑\)','😃',text)
    text = re.sub(r":\)"," ☺️ ",text)
    text = re.sub(r":-]"," ☺️ ",text)
    text = re.sub(r":]"," ☺️ ",text)
    text = re.sub(r"8\-\)"," ☺️ ",text)
    text = re.sub(r":o\)"," ☺️ ",text)
    text = re.sub(r":-}"," ☺️ ",text)
    text = re.sub(r":}"," ☺️ ",text)
    text = re.sub(r":\-\)"," ☺️ ",text)
    text = re.sub(r":c\)"," ☺️ ",text)
    text = re.sub(r":^\)"," ☺️ ",text)
    text = re.sub(r"=]"," ☺️ ",text)
    text = re.sub(r"=\)"," ☺️ ",text)
    text = re.sub(r"B^D"," 😃 ",text)
    text = re.sub(r":-\)\)"," 😃 ",text)
    text = re.sub(r":-\("," ☹️ ",text)

    text = re.sub(r":‑\("," ☹️ ",text)
    text = re.sub(r":\("," ☹️ ",text)
    text = re.sub(r":\‑\["," ☹️ ",text)
    text = re.sub(r":\["," ☹️ ",text)
    text = re.sub(r":-\|\|"," ☹️ ",text)
    text = re.sub(r">\:\["," ☹️ ",text)
    text = re.sub(r":{"," ☹️ ",text)
    text = re.sub(r">\:\("," ☹️ ",text)
    text = re.sub(r":'‑\("," 😭 ",text)
    text = re.sub(r":'\("," 😭 ",text)
    text = re.sub(r":'\‑\)"," 😃 ",text)
    text = re.sub(r":'\)"," 😃 ",text)
    text = re.sub(r":-\*"," 😗 ",text)
    text = re.sub(r":\*"," 😗 ",text)
    text = re.sub(r";\‑\)"," 😉 ",text)
    text = re.sub(r";\)"," 😉 ",text)
    text = re.sub(r"\*\-\)"," 😉 ",text)
    text = re.sub(r"\*\)"," 😉 ",text)
    text = re.sub(r";‑\]"," 😉 ",text)
    text = re.sub(r";\]"," 😉 ",text)
    text = re.sub(r";^\)"," 😉 ",text)
    text = re.sub(r">\:\[\(\)\]"," 😕 ",text)

    text = re.sub(r":\[\(\)\]"," 😕 ",text)
    text = re.sub(r"=\[\(\)\]"," 😕 ",text)
    text = re.sub(r":‑\|"," 😐 ",text)
    text = re.sub(r":\|"," 😐 ",text)
    text = re.sub(r"O:‑\)"," 😇 ",text)
    text = re.sub(r"O:\)"," 😇 ",text)
    text = re.sub(r"0:‑\)"," 😇 ",text)
    text = re.sub(r"0:\)"," 😇 ",text)
    text = re.sub(r"0;^\)"," 😇 ",text)
    text = re.sub(r">:‑\)"," 😈 ",text)
    text = re.sub(r">:\)"," 😈 ",text)
    text = re.sub(r"}:‑\)"," 😈 ",text)
    text = re.sub(r"}:\)"," 😈 ",text)
    text = re.sub(r"3:‑\)"," 😈 ",text)
    text = re.sub(r"3:\)"," 😈 ",text)
    text = re.sub(r">;\)"," 😈 ",text)
    text = re.sub(r"\|;‑\)"," 😎 ",text)
    text = re.sub(r"\|‑O"," 😏 ",text)
    text = re.sub(r"%‑\)"," 😵 ",text)
    text = re.sub(r"%\)"," 😵 ",text)
    text = re.sub(r"\(>_<\)"," 😣 ",text)
    text = re.sub(r"\(>_<\)>"," 😣 ",text)
    text = re.sub(r"\(';'\)"," Baby ",text)
    text = re.sub(r"\(^^>``"," 😓 ",text)
    text = re.sub(r"\(^_^;\)"," 😓 ",text)
    text = re.sub(r"\(-_-;\)"," 😓 ",text)

    text = re.sub(r"\(~_~;\) \(・\.・;\)"," 😓 ",text)
    text = re.sub(r"\(-_-\)zzz"," 😴 ",text)
    text = re.sub(r"\(^_-\)"," 😉 ",text)
    text = re.sub(r"\(\(\+_\+\)\)"," 😕 ",text)
    text = re.sub(r"\(\+o\+\)"," 😕 ",text)
    text = re.sub(r"^_^"," 😃 ",text)
    text = re.sub(r"\(^_^\)/"," 😃 ",text)
    text = re.sub(r"\(^O^\)／"," 😃 ",text)
    text = re.sub(r"\(__\)"," 🙇 ",text)
    text = re.sub(r"_\(._.\)_"," 🙇 ",text)
    text = re.sub(r"<\(_ _\)>"," 🙇 ",text)
    text = re.sub(r"<m\(__\)m>"," 🙇 ",text)
    text = re.sub(r"m\(__\)m"," 🙇 ",text)
    text = re.sub(r"m\(_ _\)m"," 🙇 ",text)
    text = re.sub(r"\('_'\)"," 😭 ",text)
    text = re.sub(r"\(/_;\)"," 😭 ",text)
    text = re.sub(r"\(T_T\) (;_;)"," 😭 ",text)
    text = re.sub(r"\(;_;"," 😭 ",text)
    text = re.sub(r"\(;_:\)"," 😭 ",text)
    text = re.sub(r"\(;O;\)"," 😭 ",text)
    text = re.sub(r"\(:_;\)"," 😭 ",text)
    text = re.sub(r"\(ToT\)","  😭  ",text)
    text = re.sub(r"Q\.Q"," 😭 ",text)
    text = re.sub(r"T\.T"," 😭 ",text)
    text = re.sub(r"\(-\.-\)"," 😞 ",text)
    text = re.sub(r"\(-_-\)"," 😞 ",text)

    text = re.sub(r"\(一一\)"," 😞 ",text)
    text = re.sub(r"\(；一_一\)"," 😞 ",text)

    text = re.sub(r"\(=\_=\)"," 😩 ",text)
    text = re.sub(r"\(=^\·^=\)"," 😺 ",text)

    text = re.sub(r"\(=^··^=\)"," 😺 ",text)
    text = re.sub(r"=_^= "," 😺 ",text)

    text = re.sub(r"\(\.\.\)"," 😔 ",text)

    text = re.sub(r"\(\._\.\)"," 😔  ",text)
    text = re.sub(r"\(・・\?"," 😕 ",text)
    text = re.sub(r"\(\?_\?\)"," 😕 ",text)

    text = re.sub(r">^_^<"," 😃 ",text)
    text = re.sub(r"<^\!^>"," 😃 ",text)
    text = re.sub(r"^/^","  😃  ",text)
    text = re.sub(r"\(\*^_^\*\)","  😃  ",text)
    text = re.sub(r"\(^^\)","  😃  ",text)
    text = re.sub(r"\(^\.^\)","  😃  ",text)
    text = re.sub(r"\(^_^\.\)","  😃  ",text)
    text = re.sub(r"\(^_^\)","  😃  ",text)
    text = re.sub(r"\(^J^\)","  😃  ",text)
    text = re.sub(r"\(\*^\.^\*\)"," 😃  ",text)
    text = re.sub(r"\(^—^\）"," 😃 ",text)

    text = re.sub(r"\(#^.^#\)"," 😃 ",text)
    text = re.sub(r"\(^—^\)"," 👋 ",text)
    text = re.sub(r"\(;_;\)/~~~","  👋  ",text)

    text = re.sub(r"\(^.^\)/~~~"," 👋 ",text)
    text = re.sub(r"\(-_-\)/~~~ \($··\)/~~~"," 👋 ",text)
    text = re.sub(r"\(T_T\)/~~~"," 👋 ",text)

    text = re.sub(r"\(ToT\)/~~~"," 👋 ",text)
    text = re.sub(r"\(\*^0^\*\)"," 😍 ",text)
    text = re.sub(r"\(\*_\*\)"," 😍 ",text)

    text = re.sub(r"\(\*_\*;"," 😍 ",text)
    text = re.sub(r"\(+_+\) \(@_@\)"," 😍 ",text)
    text = re.sub(r"o\.O"," 😮 ",text)
    text = re.sub(r"\(o\.o\)"," 😮 ",text)

    return text
