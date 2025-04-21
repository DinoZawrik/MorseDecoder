import torch
from config import config # Импортируем BLANK_CHAR

def decode_predictions(log_probs: torch.Tensor, int_to_char_map: dict, blank_idx: int) -> list[str]:
    """ Простое жадное CTC-декодирование (Greedy Best Path) """
    # ... (код функции без изменений) ...
    decoded_preds = []
    if log_probs.numel() == 0: return []
    log_probs_cpu = log_probs.cpu().detach()
    # Убедимся, что log_probs имеет форму [Time, Batch, Classes]
    if log_probs_cpu.dim() != 3:
        raise ValueError(f"Expected log_probs to have 3 dimensions [T, B, C], but got {log_probs_cpu.dim()}")

    best_paths = torch.argmax(log_probs_cpu, dim=2) # [Time, Batch]
    for i in range(best_paths.shape[1]): # Iterate over batch
        best_path_i = best_paths[:, i].tolist()
        decoded_seq = []
        last_char_idx = blank_idx
        for char_idx in best_path_i:
            if char_idx != blank_idx and char_idx != last_char_idx:
                decoded_seq.append(int_to_char_map.get(char_idx, '?')) # '?' если индекс не найден
            last_char_idx = char_idx
        decoded_preds.append("".join(decoded_seq))
    return decoded_preds


def decode_targets(targets: torch.Tensor, target_lengths: torch.Tensor, int_to_char_map: dict) -> list[str]:
    """ Декодирует батч таргетов (конкатенированных) из индексов в строки """
    # ... (код функции без изменений) ...
    decoded_targets_list = []
    start_idx = 0
    targets_cpu = targets.cpu().tolist()
    target_lengths_cpu = target_lengths.cpu().tolist()
    for length in target_lengths_cpu:
        if length == 0:
            decoded_targets_list.append("")
            continue
        target_indices = targets_cpu[start_idx : start_idx + length]
        decoded_target = "".join([int_to_char_map.get(idx, '?') for idx in target_indices])
        decoded_targets_list.append(decoded_target)
        start_idx += length
    return decoded_targets_list