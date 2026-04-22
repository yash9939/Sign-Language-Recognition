def update_sentence(sentence, pred_label):
    
<<<<<<< HEAD
    if pred_label == "nothing":
        return sentence

    elif pred_label == "space":
=======
    if pred_label == "" or pred_label == "nothing":
        return sentence

    elif pred_label == "space":
        if len(sentence) == 0 or sentence[-1] == " ":
            return sentence
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253
        return sentence + " "

    elif pred_label == "del":
        return sentence[:-1]

    else:
<<<<<<< HEAD
        # Prevent repeating same letter continuously ... unless it's a space
=======
<<<<<<< HEAD
=======
        # Prevent repeating same letter continuously
>>>>>>> 65ba29d3f8e1316bfc9b362047c694bf363a8ec2
        if len(sentence) > 0 and sentence[-1] == pred_label:
            return sentence
>>>>>>> e7f94120b8e68fc0d16059433aa44b447e4ec253
        return sentence + pred_label