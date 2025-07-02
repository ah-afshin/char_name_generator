import torch as t
from torch import nn

from models import NameGeneratorLSTM


def _load_dataset(path: str = "./data/names/English.txt") -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")
        # print(names[:10])
        return names


def _get_letter_dict(names: list[str]) -> tuple[dict, dict, int]:
    all_letters = sorted(list(set(''.join(names))))
    all_letters.append("<EOS>") # end of sequence
    indx_to_char = {i: ch for i, ch in enumerate(all_letters)}
    char_to_indx = {ch: i for i, ch in enumerate(all_letters)}
    return indx_to_char, char_to_indx, len(all_letters)


def _name_to_tensors(name: str, char_to_indx: dict[str, int]) -> tuple[t.Tensor, t.Tensor]:
    indices = [char_to_indx[ch] for ch in name] + [char_to_indx["<EOS>"]]
    return t.tensor(indices[:-1]), t.tensor(indices[1:])


def get_data():
    names = _load_dataset()
    indx_to_char, char_to_indx, n_letters = _get_letter_dict(names)
    
    data = []
    for name in names:
        data.append(_name_to_tensors(name, char_to_indx))
    
    # we can't stack these tensors (like `input_tensors` and `target_tensors`) 'cuz they're in different sizes
    return data, indx_to_char, char_to_indx, n_letters
    

def _train_single(model, optim, loss_func, input_data, target, n_letter) -> t.Tensor:
    one_hot_input = t.eye(n_letter)[input_data].squeeze() # we have 53 letters in this dataset
    one_hot_input = one_hot_input.unsqueeze(1) # [seq_len, B=1, inp_size]
    # print("input:", input_data.shape)            # expected: [seq_len]
    # print("one_hot:", one_hot_input.shape)       # expected: [seq_len, 1, 53]

    out, _ = model(one_hot_input)
    loss = loss_func(out, target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def train(model: nn.Module, epochs: int) -> None:
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optim = t.optim.Adam(model.parameters(), lr=0.005)

    data, indx_to_char, char_to_indx, n_letters = get_data()
    for epoch in range(epochs):
        epoch_loss = 0
        for input_data, target in data: # no batching, B = 1
            loss = _train_single(model, optim, loss_func, input_data, target, n_letters)
            if epoch%20 == 19:
                epoch_loss += loss
        if epoch%20 == 19:
            print(f"Epoch {epoch+1} | loss: {epoch_loss/len(data)}")


def main():
    # data, indx_to_char, char_to_indx, n_letters = get_data()
    # hidden_size = 128
    # model = NameRNN(input_size=n_letters, hidden_size=hidden_size, output_size=n_letters)

    model = NameGeneratorLSTM()
    train(model, epochs=100)

    t.save(model.state_dict(), "./models/model_0.pth")
    print("Model saved as model.pth")


if __name__=="__main__":
    main()
