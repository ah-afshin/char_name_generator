import torch as t
from torch import nn

from train import get_data
from models import NameGeneratorLSTM



def generate(
        input: str,
        model: nn.Module,
        indx_to_char: dict[int, str],
        char_to_indx: dict[str, int],
        n_letters: int = 53,
        max_len: int = 15
    ) -> str:
    model.eval()

    h_c = None # for first time in the loop
    pred_chars = []
    one_hot = t.eye(n_letters)[[char_to_indx[i] for i in input]].unsqueeze(1)

    for i in range(max_len):
        with t.no_grad():
            logits, h_c = model(one_hot, h_c)
            pred_prob = t.softmax(logits, dim=1)
            pred_label = t.multinomial(pred_prob, 1)
            output_char = indx_to_char[pred_label[0].item()]
            
            if output_char=="<EOS>": # end of sequence
                break

            one_hot = t.eye(n_letters)[
                                        [char_to_indx[output_char]]
                                    ].unsqueeze(1)
            pred_chars.append(output_char)
            
    pred_text = ''.join(pred_chars)
    return pred_text


if __name__ == "__main__":
    model_0 = NameGeneratorLSTM()
    model_0.load_state_dict(t.load(f="./models/model_0.pth"))
    _, indx_to_char, char_to_indx, n_letters = get_data()
    inp = input("input: ")
    result = generate(inp, model_0, indx_to_char, char_to_indx, n_letters, 15)
    print(result)
