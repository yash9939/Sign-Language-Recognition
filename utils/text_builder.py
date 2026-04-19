def update_sentence(sentence, pred_label):
    
    if pred_label == "nothing":
        return sentence

    elif pred_label == "space":
        return sentence + " "

    elif pred_label == "del":
        return sentence[:-1]

    else:
        return sentence + pred_label