# models/utils.py
from typing import Union, Tuple, List

# Dizionario per convertire numeri in parole (fino a 1000, estendibile se necessario)
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
    """
    Converte un numero nella sua parola inglese corrispondente.
    Es. 1 -> "one", 21 -> "twenty-one"
    """
    num = str(int(num))
    return num_to_word.get(num, num)

def format_count(count: Union[float, Tuple[float, float]], prompt_type: str = "word") -> str:
    """
    Genera il prompt testuale basato sul conteggio o sul range (bin).
    
    Args:
        count: Un numero singolo (es. 5) o una tupla [min, max] (es. [11, 12]).
        prompt_type: "word" (usa parole, es. 'five') o "number" (usa cifre, es. '5').
    """
    # Caso 0: Nessuna persona
    if count == 0 or (isinstance(count, (list, tuple)) and count == [0, 0]):
        return "There is no person." if prompt_type == "word" else "There is 0 person."
    
    # Caso 1: Una persona
    elif count == 1 or (isinstance(count, (list, tuple)) and count == [1, 1]):
        return "There is one person." if prompt_type == "word" else "There is 1 person."
    
    # Caso Numero Singolo (es. 5)
    elif isinstance(count, (int, float)):
        val = int(count)
        word = num2word(val) if prompt_type == "word" else str(val)
        return f"There are {word} people."
    
    # Caso Infinito / Ultimo Bin (es. [15, 9999])
    elif isinstance(count, (list, tuple)) and (count[1] == float("inf") or count[1] > 1000):
        val = int(count[0])
        word = num2word(val) if prompt_type == "word" else str(val)
        return f"There are more than {word} people."
    
    # Caso Range (es. [11, 12])
    else:  
        left, right = int(count[0]), int(count[1])
        if left == right: # Caso [5, 5] gestito come numero singolo
            word = num2word(left) if prompt_type == "word" else str(left)
            return f"There are {word} people."
        
        w_left = num2word(left) if prompt_type == "word" else str(left)
        w_right = num2word(right) if prompt_type == "word" else str(right)
        return f"There are between {w_left} and {w_right} people."

def get_prompts_from_bins(bins: List[Tuple[float, float]], prompt_type: str = "word") -> List[str]:
    """Helper per generare una lista di prompt dai bin di configurazione."""
    return [format_count(b, prompt_type) for b in bins]