# models/clip/utils.py
from typing import Union, Tuple, List

# Dizionario per convertire numeri in parole
num_to_word = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", 
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten", 
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen", "15": "fifteen", 
    "16": "sixteen", "17": "seventeen", "18": "eighteen", "19": "nineteen", 
    "20": "twenty", "21": "twenty-one", "22": "twenty-two", "23": "twenty-three", 
    "24": "twenty-four", "25": "twenty-five", "26": "twenty-six", "27": "twenty-seven", 
    "28": "twenty-eight", "29": "twenty-nine", "30": "thirty", "40": "forty", 
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
    "100": "one hundred"
}

def num2word(num: Union[int, str]) -> str:
    """Convert number to English word."""
    num = str(int(num))
    return num_to_word.get(num, num)

def format_count(count: Union[float, Tuple[float, float]], prompt_type: str = "word") -> str:
    """
    Genera il prompt testuale basato sul conteggio.
    Es: [0,0] -> "There is 0 person."
        [4, inf] -> "There are more than 4 people."
    """
    # Caso 1: Numero esatto (0, 1, 2...) rappresentato come float o [x, x]
    if isinstance(count, (int, float)):
        val = int(count)
        is_exact = True
    elif isinstance(count, (list, tuple)) and count[0] == count[1]:
        val = int(count[0])
        is_exact = True
    else:
        is_exact = False

    if is_exact:
        if val == 0:
            return "There is no person." if prompt_type == "word" else "There is 0 person."
        elif val == 1:
            return "There is one person." if prompt_type == "word" else "There is 1 person."
        else:
            term = num2word(val) if prompt_type == "word" else str(val)
            return f"There are {term} people."

    # Caso 2: Intervallo o Infinito
    # Assumiamo count sia una tupla/lista [min, max]
    start, end = count[0], count[1]
    
    if end == float("inf") or str(end) == "inf":
        term = num2word(start) if prompt_type == "word" else str(int(start))
        return f"There are more than {term} people."
    else:
        # Range finito [a, b]
        if prompt_type == "word":
            s_term = num2word(start)
            e_term = num2word(end)
        else:
            s_term = str(int(start))
            e_term = str(int(end))
        return f"There are between {s_term} and {e_term} people."