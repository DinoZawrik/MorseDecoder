import random
import os
import numpy as np
import torch

def seed_everything(seed):
    """Устанавливает seed для воспроизводимости."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Установка benchmark=False и deterministic=True может замедлить обучение
        # но обеспечивает большую воспроизводимость. Оставьте False/True для скорости.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def check_ctc_decoder():
    """Проверяет доступность CTCDecoder."""
    try:
        from torchaudio.models.decoder import ctc_decoder
        print("Фабричная функция `ctc_decoder` импортирована.")
        return True, ctc_decoder
    except ImportError:
        print("Предупреждение: `torchaudio.models.decoder.ctc_decoder` не найден.")
        print("Для Beam Search установите: !pip install flashlight-text --no-deps")
        print("!!! И ПЕРЕЗАПУСТИТЕ СРЕДУ ВЫПОЛНЕНИЯ !!!")
        return False, None

# Можно добавить функцию для настройки logging, если нужно