import editdistance
try:
    import jiwer
    JIWER_AVAILABLE = True
except ImportError:
    JIWER_AVAILABLE = False
    print("Warning: jiwer not installed. CER calculation will be skipped. Install with: pip install jiwer")


def calculate_levenshtein_mean(predictions: list[str], targets: list[str]) -> float:
    """ Расчет среднего расстояния Левенштейна """
    # ... (код функции без изменений) ...
    if not predictions or not targets or len(predictions) != len(targets):
        # print("Warning: Mismatched predictions/targets length for Levenshtein.")
        return float('inf')
    total_distance, num_samples = 0, len(targets)
    if num_samples == 0: return 0.0
    for pred, target in zip(predictions, targets):
        try:
            total_distance += editdistance.eval(str(pred), str(target))
        except Exception as e:
            print(f"Error in editdistance.eval for pred='{pred}', target='{target}': {e}")
            total_distance += len(target) # Penalize with target length
    return total_distance / num_samples


def calculate_cer(predictions: list[str], targets: list[str]) -> float:
    """ Расчет среднего CER с помощью jiwer """
    if not JIWER_AVAILABLE: return float('nan')
    # ... (код функции без изменений) ...
    if not predictions or not targets or len(predictions) != len(targets):
        # print("Warning: Mismatched predictions/targets length for CER.")
        return float('inf')

    try:
        # jiwer может падать на пустых таргетах или предсказаниях, фильтруем
        filtered_preds = []
        filtered_targets = []
        for p, t in zip(predictions, targets):
             # jiwer может не любить пустые строки, зависит от версии
             # Допустим, что пустой таргет и пустой пред - это 0 ошибки
             if not t and not p: continue
             # Если таргет пустой, а пред нет - ошибка 100% для этого сэмпла (или inf?)
             # Jiwer обычно обрабатывает это, но добавим защиту
             if not t and p:
                  filtered_preds.append(p)
                  filtered_targets.append(" ") # Добавляем пробел, чтобы jiwer не упал
             else:
                 filtered_preds.append(p if p else " ") # Заменяем пустой пред пробелом
                 filtered_targets.append(t)

        if not filtered_targets: return 0.0 # Нет валидных пар для сравнения

        # Используем ground_truth, hypothesis порядок для jiwer.cer
        return jiwer.cer(filtered_targets, filtered_preds)
    except Exception as e:
        print(f"Ошибка расчета CER с помощью jiwer: {e}")
        return float('nan')