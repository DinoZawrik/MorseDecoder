import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.fft 

import pandas as pd
import numpy as np
import os
import time
import random
import traceback
import json
from tqdm import tqdm 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


from config import config 
from utils.helpers import seed_everything, check_ctc_decoder
from utils.decoding import decode_predictions, decode_targets
from utils.metrics import calculate_levenshtein_mean, calculate_cer
from data.dataset import MorseDataset
from data.collate import collate_fn
from models.crnn import CRNNModel_4Layer 

def run_training():
    print(f"\n--- Запуск Обучения: {config.MODEL_NAME} ---")
    start_time_total = time.time()

    seed_everything(config.SEED)
    DEVICE = config.DEVICE
    TORCHAUDIO_CTC_DECODER_AVAILABLE, _ = check_ctc_decoder()

    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print(f"Device: {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"Предобработка: Normalize={config.PREPROCESSING['normalize']['apply']}, FreqFilter={config.PREPROCESSING['frequency_filter']['apply']}")
    print(f"Аугментации (Prob={config.AUGMENTATION['probability']}): {config.AUGMENTATION.get('active', [])}")
    print(f"Модель: {config.MODEL['type']}")
    print(f"Обучение: {config.TRAINING['num_epochs']} эпох, BS={config.TRAINING['batch_size']}, LR={config.TRAINING['learning_rate']}")
    print(f"Чекпоинты: {config.CHECKPOINT_DIR}")
    print(f"Логи: {config.LOG_FILE}")

    print("\nЗагрузка метаданных...")
    try:
        train_df_full = pd.read_csv(config.TRAIN_CSV_PATH)
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл {config.TRAIN_CSV_PATH}. Убедитесь, что данные доступны.")
        raise e
    train_df_full[config.TARGET_COLUMN] = train_df_full[config.TARGET_COLUMN].fillna('').astype(str)
    # Фильтруем пустые таргеты ДО сплита
    train_df_filtered = train_df_full[train_df_full[config.TARGET_COLUMN].str.len() > 0].copy()
    print(f"Записей для обучения (после фильтрации пустых): {len(train_df_filtered)}")
    if len(train_df_filtered) == 0:
        raise ValueError("Нет валидных данных для обучения после фильтрации!")

    all_chars = set(train_df_filtered[config.TARGET_COLUMN].str.cat(sep=''))
    chars = [config.BLANK_CHAR] + sorted(list(all_chars))
    char_to_int = {char: i for i, char in enumerate(chars)}
    int_to_char = {i: char for i, char in enumerate(chars)}
    NUM_CLASSES = len(chars)
    blank_index = char_to_int[config.BLANK_CHAR]
    print(f"Алфавит создан ({NUM_CLASSES} классов). Blank: '{config.BLANK_CHAR}' ({blank_index})")
    print(f"Символы ({len(chars)}): {''.join(chars)}")

    print("\nРазделение данных и создание датасетов/лоадеров...")
    train_df, val_df = train_test_split(
        train_df_filtered,
        test_size=config.TRAINING['validation_split_size'],
        random_state=config.SEED
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print(f"Размеры выборок: Train={len(train_df)}, Validation={len(val_df)}")

    train_dataset = MorseDataset(
        data_frame=train_df, audio_base_path=config.AUDIO_BASE_PATH, char_map=char_to_int,
        file_path_column=config.FILE_PATH_COLUMN, target_column=config.TARGET_COLUMN,
        test_id_column=config.TEST_ID_COLUMN,
        audio_cfg=config.AUDIO, preproc_cfg=config.PREPROCESSING, aug_cfg=config.AUGMENTATION,
        is_train=True
    )
    val_dataset = MorseDataset(
        data_frame=val_df, audio_base_path=config.AUDIO_BASE_PATH, char_map=char_to_int,
        file_path_column=config.FILE_PATH_COLUMN, target_column=config.TARGET_COLUMN,
        test_id_column=config.TEST_ID_COLUMN,
        audio_cfg=config.AUDIO, preproc_cfg=config.PREPROCESSING, aug_cfg=None,
        is_train=False
    )
    print("Датасеты созданы.")


    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAINING['batch_size'], shuffle=True,
        collate_fn=collate_fn, num_workers=config.TRAINING['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.TRAINING['num_workers'] > 0),
        drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.INFERENCE['batch_size'], shuffle=False,
        collate_fn=collate_fn, num_workers=config.TRAINING['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(config.TRAINING['num_workers'] > 0)
    )
    print("Лоадеры созданы.")


    print("\nНастройка компонентов обучения...")
    if config.MODEL['type'] == 'CRNNModel_4Layer':
        model = CRNNModel_4Layer(
            n_features=config.AUDIO['n_mels'], num_classes=NUM_CLASSES,
            rnn_hidden_size=config.MODEL['rnn_hidden_size'], num_rnn_layers=config.MODEL['rnn_num_layers'],
            cnn_dropout=config.MODEL['cnn_dropout'], rnn_dropout=config.MODEL['rnn_dropout']
        ).to(DEVICE)
    else:
        raise ValueError(f"Неизвестный тип модели в конфиге: {config.MODEL['type']}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Модель {config.MODEL['type']} создана ({num_params:,} параметров) на {DEVICE}.")


    criterion = nn.CTCLoss(blank=blank_index, reduction='mean', zero_infinity=True).to(DEVICE)

    # Оптимизатор
    lr = config.TRAINING['learning_rate']
    wd = config.TRAINING['optimizer_weight_decay']
    if config.TRAINING['optimizer'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif config.TRAINING['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Неизвестный оптимизатор: {config.TRAINING['optimizer']}")
    print(f"Оптимизатор: {config.TRAINING['optimizer']} (LR={lr}, WD={wd})")

    scheduler = None
    scheduler_name = config.TRAINING.get('scheduler', 'None') 
    if scheduler_name == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                                      factor=config.TRAINING['scheduler_factor'],
                                      patience=config.TRAINING['scheduler_patience'],
                                      verbose=True)
    elif scheduler_name == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.TRAINING['num_epochs'], eta_min=lr / 100)
    elif scheduler_name is not None and scheduler_name != 'None':
        raise ValueError(f"Неизвестный планировщик: {scheduler_name}")
    print(f"Планировщик: {scheduler_name if scheduler else 'None'}")

    # --- 4. Цикл Обучения ---
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_levenshtein': [], 'val_cer': [], 'lr': []}
    best_val_metric = float('inf')
    best_epoch = -1


    log_header = "Epoch,Train Loss,Val Loss,Val Levenshtein,Val CER,Time(s),LR\n"
    try:
        with open(config.LOG_FILE, "w") as f:
            f.write(log_header)
        print(f"Лог файл инициализирован: {config.LOG_FILE}")
    except IOError as e:
        print(f"Ошибка записи в лог файл {config.LOG_FILE}: {e}")


    print(f"\n--- Старт обучения ({config.TRAINING['num_epochs']} эпох) ---")

    for epoch in range(config.TRAINING['num_epochs']):
        epoch_start_time = time.time()
        model.train()
        train_loss_epoch = 0.0
        train_batches = 0
        train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.TRAINING['num_epochs']} [Train]", leave=False)

        for batch_idx, batch in enumerate(train_iter):
            if not isinstance(batch, (tuple, list)) or len(batch) != 4:
                 continue
            spec_batch, spec_len_batch, target_batch, target_len_batch = batch
            if spec_batch.numel() == 0 or target_batch.numel() == 0:

                 continue

            spec_batch = spec_batch.to(DEVICE)
            spec_len_batch = spec_len_batch.to(DEVICE)
            target_batch = target_batch.to(DEVICE)
            target_len_batch = target_len_batch.to(DEVICE)

            optimizer.zero_grad()

            try:

                log_probs, output_lengths = model(spec_batch, spec_len_batch)

                output_lengths = torch.clamp(output_lengths, max=log_probs.shape[0])

                loss = criterion(log_probs, target_batch, output_lengths.cpu(), target_len_batch.cpu())

                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\nПредупреждение: NaN/inf loss на трейне (эпоха {epoch+1}, батч {batch_idx}). Пропуск батча.")
                    continue

                loss.backward()

                if config.TRAINING['grad_clip_value'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAINING['grad_clip_value'])

                optimizer.step()

                train_loss_epoch += loss.item()
                train_batches += 1

                if batch_idx % 20 == 0 or batch_idx == len(train_iter) - 1:
                    train_iter.set_postfix(loss=f"{train_loss_epoch / train_batches:.4f}")

            except Exception as e:
                print(f"\nОшибка на шаге обучения (эпоха {epoch+1}, батч {batch_idx}): {e}")
                print(f"  Shapes: spec={spec_batch.shape}, spec_len={spec_len_batch.shape}, target={target_batch.shape}, target_len={target_len_batch.shape}")

                continue

        avg_train_loss = train_loss_epoch / train_batches if train_batches > 0 else float('nan')

        model.eval()
        val_loss_epoch = 0.0
        val_batches = 0
        all_val_preds, all_val_targets = [], []
        val_iter = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.TRAINING['num_epochs']} [Val]", leave=False)

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_iter):
                if not isinstance(batch, (tuple, list)) or len(batch) != 4: continue
                spec_batch, spec_len_batch, target_batch, target_len_batch = batch
                if spec_batch.numel() == 0 or target_batch.numel() == 0: continue

                spec_batch = spec_batch.to(DEVICE)
                spec_len_batch = spec_len_batch.to(DEVICE)
                target_batch = target_batch.to(DEVICE)
                target_len_batch = target_len_batch.to(DEVICE)

                try:
                    log_probs, output_lengths = model(spec_batch, spec_len_batch)
                    output_lengths = torch.clamp(output_lengths, max=log_probs.shape[0])

                    loss = criterion(log_probs, target_batch, output_lengths.cpu(), target_len_batch.cpu())

                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss_epoch += loss.item()
                        val_batches += 1
                    else:
                         print(f"\nПредупреждение: NaN/inf loss на валидации (эпоха {epoch+1}, батч {batch_idx}).")


                    batch_preds = decode_predictions(log_probs, int_to_char, blank_index)
                    batch_targets = decode_targets(target_batch, target_len_batch, int_to_char)
                    all_val_preds.extend(batch_preds)
                    all_val_targets.extend(batch_targets)

                except Exception as e:
                    print(f"\nОшибка на шаге валидации (эпоха {epoch+1}, батч {batch_idx}): {e}")
                    print(f"  Shapes: spec={spec_batch.shape}, spec_len={spec_len_batch.shape}, target={target_batch.shape}, target_len={target_len_batch.shape}")
                    continue 

        avg_val_loss = val_loss_epoch / val_batches if val_batches > 0 else float('nan')
        val_levenshtein = calculate_levenshtein_mean(all_val_preds, all_val_targets)
        val_cer = calculate_cer(all_val_preds, all_val_targets)

        current_lr = optimizer.param_groups[0]['lr']
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_levenshtein'].append(val_levenshtein)
        history['val_cer'].append(val_cer)
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start_time

        log_str = (f"E{epoch+1:02d} [{epoch_time:.1f}s] | "
                   f"TrL={avg_train_loss:.4f} | VL={avg_val_loss:.4f} | "
                   f"VLev={val_levenshtein:.4f} | VCER={val_cer:.4f} | "
                   f"LR={current_lr:.1e}")
        print(log_str)
        try:
            with open(config.LOG_FILE, "a") as f:
                f.write(f"{epoch+1},{avg_train_loss:.4f},{avg_val_loss:.4f},{val_levenshtein:.4f},{val_cer:.4f},{epoch_time:.1f},{current_lr:.1e}\n")
        except IOError as e:
            print(f"Ошибка записи лога эпохи {epoch+1} в файл: {e}")


        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_levenshtein) 
            elif isinstance(scheduler, CosineAnnealingLR):
                scheduler.step()

        if val_levenshtein < best_val_metric:
            best_val_metric = val_levenshtein
            best_epoch = epoch + 1
            checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{config.MODEL_NAME}_best.pth")

            try:
                save_dict = {
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_metric': best_val_metric, 
                    'char_to_int': char_to_int,   
                    'int_to_char': int_to_char,   
                    'blank_index': blank_index,   
                    'config': { 
                        'AUDIO': config.AUDIO,
                        'PREPROCESSING': config.PREPROCESSING,
                        'MODEL': config.MODEL,
                        'MODEL_NAME': config.MODEL_NAME,
                        'TARGET_COLUMN': config.TARGET_COLUMN,
                        'FILE_PATH_COLUMN': config.FILE_PATH_COLUMN,
                        'TEST_ID_COLUMN': config.TEST_ID_COLUMN
                    }
                }
                torch.save(save_dict, checkpoint_path)
                print(f"  -> Best model saved to: {checkpoint_path} (Epoch: {best_epoch}, Levenshtein: {best_val_metric:.4f})")

            except Exception as e:
                print(f"Ошибка сохранения чекпоинта: {e}")

    total_training_time = time.time() - start_time_total
    print(f"\n--- Обучение завершено за {total_training_time/60:.1f} мин ---")
    if best_epoch != -1:
        print(f"Лучшая модель сохранена (Эпоха {best_epoch}) с Val Levenshtein: {best_val_metric:.4f}")
    else:
        print("Лучшая модель не была сохранена (возможно, валидация не улучшалась).")

    return history

if __name__ == "__main__":
    history = run_training()

    if history and history['epoch']:
        print("\n--- Анализ Результатов Обучения ---")
        epochs_range = history['epoch']

        plt.figure(figsize=(18, 5))

        plt.subplot(1, 4, 1)
        plt.plot(epochs_range, history['train_loss'], label='Train Loss')
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
        plt.legend(loc='best'); plt.title('Loss'); plt.xlabel('Epoch'); plt.grid(True)

        plt.subplot(1, 4, 2)
        plt.plot(epochs_range, history['val_levenshtein'], label='Validation Levenshtein')
        plt.legend(loc='best'); plt.title('Levenshtein Distance'); plt.xlabel('Epoch'); plt.grid(True)

        plt.subplot(1, 4, 3)
        if not all(np.isnan(history['val_cer'])):
             plt.plot(epochs_range, history['val_cer'], label='Validation CER')
             plt.legend(loc='best'); plt.title('Character Error Rate (CER)'); plt.xlabel('Epoch'); plt.grid(True)
        else:
             plt.text(0.5, 0.5, 'CER not available\n(jiwer not installed?)', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
             plt.title('Character Error Rate (CER)')

        plt.subplot(1, 4, 4)
        plt.plot(epochs_range, history['lr'], label='Learning Rate')
        plt.legend(loc='best'); plt.title('Learning Rate'); plt.xlabel('Epoch'); plt.grid(True)


        plt.tight_layout()
        plot_filename = os.path.join(config.LOG_DIR, f"training_curves_{config.MODEL_NAME}.png")
        try:
            plt.savefig(plot_filename)
            print(f"Графики обучения сохранены: {plot_filename}")
        except Exception as e:
            print(f"Ошибка сохранения графиков: {e}")
    else:
        print("Нет данных истории для построения графиков.")