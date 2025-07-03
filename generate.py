import torch as t
from torch import nn
from treelib import Tree

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


def build_tree_with_treelib(model, start_letter, char_to_indx, indx_to_char, n_letters, max_depth=3, topk=3):
    model.eval()
    tree = Tree()
    root = start_letter
    tree.create_node(root, root)  # tag, identifier

    def recurse(current_seq, h_c, parent_id, depth):
        if depth >= max_depth:
            return
        input_tensor = t.eye(n_letters)[[char_to_indx[current_seq[-1]]]].unsqueeze(1)
        with t.no_grad():
            logits, h_c_next = model(input_tensor, h_c)
            probs = t.softmax(logits, dim=1).squeeze()
            topk_probs, topk_indices = t.topk(probs, topk)

        for i in range(topk):
            char = indx_to_char[topk_indices[i].item()]
            node_id = parent_id + char

            if char == "<EOS>":
                tree.create_node(f'"{current_seq}"', node_id, parent=parent_id)
            else:
                tree.create_node(char, node_id, parent=parent_id)
                recurse(current_seq + char, h_c_next, node_id, depth + 1)

    recurse(start_letter, None, root, 0)
    return tree


def main():
    model_0 = NameGeneratorLSTM()
    model_0.load_state_dict(t.load(f="./models/model_0.pth"))
    _, indx_to_char, char_to_indx, n_letters = get_data()
    inp = input("input: ")
    tree = build_tree_with_treelib(model_0, inp, char_to_indx, indx_to_char, n_letters, max_depth=7, topk=2)
    tree.show()


if __name__ == "__main__":
    main()
