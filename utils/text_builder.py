def update_sentence(sentence, pred_label):
    
    if pred_label == "" or pred_label == "nothing":
        return sentence

    elif pred_label == "space":
        if len(sentence) == 0 or sentence[-1] == " ":
            return sentence
        return sentence + " "

    elif pred_label == "del":
        return sentence[:-1]

    else:
        # Prevent repeating same letter continuously ... unless it's a space
        if len(sentence) > 0 and sentence[-1] == pred_label:
            return sentence
        return sentence + pred_label