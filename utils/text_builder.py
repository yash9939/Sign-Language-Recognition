def update_sentence(sentence, pred_label):

    # Ignore empty / nothing
    if pred_label == "" or pred_label == "nothing":
        return sentence

    # Handle space
    elif pred_label == "space":
        if len(sentence) == 0 or sentence[-1] == " ":
            return sentence
        return sentence + " "

    # Handle delete
    elif pred_label == "del":
        return sentence[:-1]

    # Handle normal letters
    else:
        # Prevent repeating same letter continuously
        if len(sentence) > 0 and sentence[-1] == pred_label:
            return sentence
        return sentence + pred_label