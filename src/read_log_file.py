import matplotlib.pyplot as plt
import json


def read_log_file(log_file_path: str, plot: bool = False, verbose: bool = True) -> dict:
    """Read log file"""
    train_losses, val_losses = [], []
    num_epochs = 0
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # take only latest log
    start_ids = []
    for idx, line in enumerate(lines):
        if "***RUN STARTED***" in line:
            start_ids.append(idx)
    assert len(start_ids) > 0
    if verbose:
        print(f'{log_file_path} contains {len(start_ids)} runs.\n'
              f'Considering the latest run..')
    lines = lines[start_ids[-1]:]

    for line in lines:
        if 'Epoch: ' in line:
            num_epochs = num_epochs + 1
            line = line.strip().split()
            train_losses.append(float(line[-4]))
            val_losses.append(float(line[-1]))

    # get test loss
    test_loss_line = [line for line in lines if "test_loss" in line]
    if len(test_loss_line) > 0:
        test_loss = float(test_loss_line[0].strip().split()[-1])
    else:
        test_loss = None

    # get best val metrics
    line_idx = [idx for idx, line in enumerate(lines) if "📣 best validation metrics 📣" in line][0]
    line = lines[line_idx]
    chunk = '{' + line.split('{')[-1]
    best_val_metrics = json.loads(chunk[:-1].replace("\'", "\""))

    # get test metrics
    line_ids = [idx for idx, line in enumerate(lines) if "📣 test metrics 📣" in line]
    if len(line_ids) == 1:
        line = lines[line_ids[0]]
        chunk = '{' + line.split('{')[-1]
        test_metrics = json.loads(chunk[:-1].replace("\'", "\""))
    else:
        test_metrics = None

    # get OOV lines
    oov_lines = ''.join([line for line in lines if "running tokens are OOV" in line])

    if plot:
        # plot
        fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(6, 5), dpi=160)
        axs.plot(range(1, num_epochs + 1), train_losses, color='xkcd:orangered', label='train')
        axs.plot(range(1, num_epochs + 1), val_losses, color='xkcd:gold', label='val')
        axs.set_xlabel('Epoch', fontsize=18)
        axs.set_ylabel('Loss', fontsize=18)
        axs.legend(fontsize=18)
        plt.show()

    return {
        'num_epochs': num_epochs,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'best_val_metrics': best_val_metrics,
        'test_metrics': test_metrics,
        'oov_lines': oov_lines,
    }


if __name__ == "__main__":
    _log_file_path = '../logs/bg/transformer_encoder/few150_hidden128_vocab16000.txt'  # log filename
    print(read_log_file(log_file_path=_log_file_path, plot=True, verbose=True))
