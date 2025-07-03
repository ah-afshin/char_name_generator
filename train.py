import torch as t
from torch import nn

from model import NameGeneratorLSTM


def _load_dataset(path: str = "./data/names/English.txt") -> list[str]:
    """Load a list of names from a text file.

    Args:
        path (str): Path to the text file containing names, one per line.

    Returns:
        list[str]: List of names as strings.
    """
    with open(path, "r", encoding="utf-8") as f:
        names = f.read().strip().split("\n")
        # print(names[:10])
        return names


def _get_letter_dict(names: list[str]) -> tuple[dict, dict, int]:
    """Build mappings between characters and indices.

    Args:
        names (list[str]): List of names.

    Returns:
        tuple:
            - indx_to_char (dict[int, str]): index to character
            - char_to_indx (dict[str, int]): character to index
            - n_letters (int): total number of unique characters + EOS
    """
    all_letters = sorted(list(set(''.join(names))))
    all_letters.append("<EOS>") # end of sequence
    indx_to_char = {i: ch for i, ch in enumerate(all_letters)}
    char_to_indx = {ch: i for i, ch in enumerate(all_letters)}
    return indx_to_char, char_to_indx, len(all_letters)


def _name_to_tensors(name: str, char_to_indx: dict[str, int]) -> tuple[t.Tensor, t.Tensor]:
    """Convert a name string to input and target tensor of indices.

    Args:
        name (str): The name string.
        char_to_indx (dict[str, int]): Character to index mapping.

    Returns:
        tuple[t.Tensor, t.Tensor]: input and target tensors (as indices).
    """
    indices = [char_to_indx[ch] for ch in name] + [char_to_indx["<EOS>"]]
    return t.tensor(indices[:-1]), t.tensor(indices[1:])


def get_data():
    """Load data and prepare training pairs (input, target) for each name.

    Returns:
        tuple: 
            - data (list of (input_tensor, target_tensor)),
            - indx_to_char (dict),
            - char_to_indx (dict),
            - n_letters (int)
    """
    names = _load_dataset()
    indx_to_char, char_to_indx, n_letters = _get_letter_dict(names)
    
    data = []
    for name in names:
        data.append(_name_to_tensors(name, char_to_indx))
    
    # we can't stack these tensors (like `input_tensors` and `target_tensors`) 'cuz they're in different sizes
    return data, indx_to_char, char_to_indx, n_letters
    

def _train_single(model, optim, loss_func, input_data, target, n_letter) -> t.Tensor:
    """Train the model on a single name.

    Args:
        model (nn.Module): The LSTM model.
        optim: Optimizer.
        loss_func: Loss function.
        input_data (Tensor): Input sequence.
        target (Tensor): Target sequence.
        n_letter (int): Number of unique letters.

    Returns:
        float: Loss value for this training sample.
    """
    one_hot_input = t.eye(n_letter)[input_data].squeeze() # we have 53 letters in this dataset
    one_hot_input = one_hot_input.unsqueeze(1) # [seq_len, B=1, inp_size]

    out, _ = model(one_hot_input)
    loss = loss_func(out, target)
    optim.zero_grad()
    loss.backward()
    optim.step()
    return loss.item()


def train(model: nn.Module, epochs: int) -> None:
    """Train the model for a number of epochs on the name dataset.

    Args:
        model (nn.Module): LSTM name generator.
        epochs (int): Number of training epochs.
    """
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
    """
    Entry point to train the name generator model and save it.
    """
    model = NameGeneratorLSTM()
    train(model, epochs=100)

    t.save(model.state_dict(), "./models/model_0.pth")
    print("Model saved as model.pth")


if __name__=="__main__":
    main()
