import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import os
import time
import traceback
from tqdm import tqdm
import json


from config import config as main_config
from utils.helpers import check_ctc_decoder
from utils.decoding import decode_predictions
from data.dataset import MorseDataset
from data.collate import collate_fn_test
from models.crnn import CRNNModel_4Layer 

def run_inference(checkpoint_path=None):
    print("\n--- Запуск Инференса ---")
    inference_start_time = time.time()
    DEVICE = main_config.DEVICE
    TORCHAUDIO_CTC_DECODER_AVAILABLE, ctc_decoder_factory = check_ctc_decoder()

    inference_model = None
    test_loader_inf = None
    decoder = None 
    decoder_type = "Greedy"
    char_to_int_inf = None
    int_to_char_inf = None
    blank_index_inf = -1
    loaded_config_inf = None

    try:
        if checkpoint_path is None:
            checkpoint_path = os.path.join(main_config.CHECKPOINT_DIR, f"{main_config.MODEL_NAME}_best.pth")

        if not os.path.exists(checkpoint_path):
             available_checkpoints = [f for f in os.listdir(main_config.CHECKPOINT_DIR) if f.endswith(".pth") and main_config.MODEL_NAME in f]
             if not available_checkpoints:
                 raise FileNotFoundError(f"Чекпоинты не найдены в {main_config.CHECKPOINT_DIR} для модели {main_config.MODEL_NAME}")
             checkpoint_path = os.path.join(main_config.CHECKPOINT_DIR, available_checkpoints[0])
             print(f"Предупреждение: Файл '{main_config.MODEL_NAME}_best.pth' не найден. Используется: {os.path.basename(checkpoint_path)}")
        else:
             print(f"Используется чекпоинт: {os.path.basename(checkpoint_path)}")


        checkpoint_inf = torch.load(checkpoint_path, map_location=DEVICE)

        if 'config' not in checkpoint_inf: raise ValueError("Конфигурация не найдена в чекпоинте!")
        if 'char_to_int' not in checkpoint_inf: raise ValueError("char_to_int не найден в чекпоинте!")
        if 'int_to_char' not in checkpoint_inf: raise ValueError("int_to_char не найден в чекпоинте!")
        if 'blank_index' not in checkpoint_inf: raise ValueError("blank_index не найден в чекпоинте!")

        loaded_config_inf = checkpoint_inf['config']
        char_to_int_inf = checkpoint_inf['char_to_int']
        int_to_char_inf = checkpoint_inf['int_to_char']
        blank_index_inf = checkpoint_inf['blank_index']
        NUM_CLASSES_INF = len(char_to_int_inf)
        
        print("Параметры из загруженного чекпоинта:")
        print(f"  Модель: {loaded_config_inf.get('MODEL', {}).get('type', 'N/A')}")
        print(f"  Аудио: n_mels={loaded_config_inf.get('AUDIO', {}).get('n_mels', 'N/A')}")
        print(f"  Препроцессинг: Normalize={loaded_config_inf.get('PREPROCESSING', {}).get('normalize', {}).get('apply', 'N/A')}, FreqFilter={loaded_config_inf.get('PREPROCESSING', {}).get('frequency_filter', {}).get('apply', 'N/A')}")


        model_cfg_inf = loaded_config_inf['MODEL']
        audio_cfg_inf = loaded_config_inf['AUDIO']
        n_features_inf = audio_cfg_inf['n_mels']

        if model_cfg_inf['type'] == 'CRNNModel_4Layer':
            inference_model = CRNNModel_4Layer(
                n_features=n_features_inf, num_classes=NUM_CLASSES_INF,
                rnn_hidden_size=model_cfg_inf['rnn_hidden_size'], num_rnn_layers=model_cfg_inf['rnn_num_layers'],
                cnn_dropout=0.0, rnn_dropout=0.0
            ).to(DEVICE)
        else:
            raise ValueError(f"Неизвестный тип модели в конфиге чекпоинта: {model_cfg_inf['type']}")

        inference_model.load_state_dict(checkpoint_inf['model_state_dict'])
        inference_model.eval()
        num_params = sum(p.numel() for p in inference_model.parameters())
        print(f"Модель {model_cfg_inf['type']} для инференса загружена ({num_params:,} параметров).")


        try:
            test_df = pd.read_csv(main_config.TEST_CSV_PATH)
        except FileNotFoundError:
            print(f"Ошибка: Тестовый файл не найден {main_config.TEST_CSV_PATH}")
            raise

        test_dataset_inf = MorseDataset(
            data_frame=test_df, audio_base_path=main_config.AUDIO_BASE_PATH,
            char_map=None, 
            file_path_column=loaded_config_inf['FILE_PATH_COLUMN'],
            target_column=loaded_config_inf['TARGET_COLUMN'],     
            test_id_column=loaded_config_inf['TEST_ID_COLUMN'],    
            audio_cfg=loaded_config_inf["AUDIO"],
            preproc_cfg=loaded_config_inf["PREPROCESSING"],
            aug_cfg=None,
            is_train=False
        )

        test_loader_inf = DataLoader(
             test_dataset_inf,
             batch_size=main_config.INFERENCE['batch_size'],
             shuffle=False,
             collate_fn=collate_fn_test,
             num_workers=0 
        )
        print(f"Тестовый лоадер создан ({len(test_dataset_inf)} сэмплов).")


        use_beam_search_inf = main_config.INFERENCE['use_beam_search'] and TORCHAUDIO_CTC_DECODER_AVAILABLE

        if use_beam_search_inf:
            try:
                labels_inf = [int_to_char_inf.get(i, '?') for i in range(NUM_CLASSES_INF)]
                if labels_inf[blank_index_inf] != main_config.BLANK_CHAR:
                     print(f"Предупреждение: Символ по blank_index={blank_index_inf} ('{labels_inf[blank_index_inf]}') не совпадает с BLANK_CHAR ('{main_config.BLANK_CHAR}'). Используется символ по индексу.")
                     blank_token_for_decoder = labels_inf[blank_index_inf]
                else:
                     blank_token_for_decoder = main_config.BLANK_CHAR

                lm_path = main_config.INFERENCE.get('beam_search_lm_path')
                lexicon_path = main_config.INFERENCE.get('beam_search_lexicon_path')
                if lm_path and not os.path.exists(lm_path):
                    print(f"Предупреждение: Файл LM не найден '{lm_path}'. Beam Search будет без LM.")
                    lm_path = None
                if lexicon_path and not os.path.exists(lexicon_path):
                    print(f"Предупреждение: Файл лексикона не найден '{lexicon_path}'.")

                decoder = ctc_decoder_factory(
                    lexicon=lexicon_path,
                    tokens=labels_inf,
                    lm=lm_path,
                    nbest=1,
                    beam_size=main_config.INFERENCE['beam_search_beam_size'],
                    blank_token=blank_token_for_decoder,
                    sil_token="|",
                    unk_word="<unk>", 
                )
                decoder_type = f"Beam Search (Size={main_config.INFERENCE['beam_search_beam_size']}, LM: {'Yes' if lm_path else 'No'}, Lexicon: {'Yes' if lexicon_path else 'No'})"
                print(f"Декодер {decoder_type} создан.")
            except Exception as e:
                print(f"Ошибка создания CTCDecoder: {e}. Используется Greedy декодирование.")
                decoder = None
                decoder_type = "Greedy (Beam Search fallback)"
        else:
            print("Beam Search отключен или недоступен. Используется Greedy декодирование.")
            decoder = None 
            decoder_type = "Greedy"

    except Exception as e:
        print(f"\nКритическая ошибка подготовки инференса: {e}")
        traceback.print_exc()
        return None

    submission_results = []
    print(f"\n--- Начало инференса ({decoder_type}) ---")
    test_iter = tqdm(test_loader_inf, desc=f"Inference")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_iter):
            if not isinstance(batch, (tuple, list)) or len(batch) != 3: continue
            spec_batch, spec_len_batch, ids_batch = batch
            if spec_batch.numel() == 0: continue

            spec_batch = spec_batch.to(DEVICE)
            spec_len_batch = spec_len_batch.to(DEVICE) 

            try:
                log_probs, output_lengths = inference_model(spec_batch, spec_len_batch)

                batch_preds = []

                if decoder:
                    emissions_p = log_probs.permute(1, 0, 2).cpu().float().contiguous()
                    lengths_c = output_lengths.cpu()

                    hypotheses = decoder(emissions_p, lengths_c)

                    for hyps in hypotheses:
                        if hyps:
                            pred_indices = hyps[0].tokens.tolist()
                            pred_str = "".join([int_to_char_inf.get(idx, '?') for idx in pred_indices])
                            batch_preds.append(pred_str.strip()) 
                        else:
                            batch_preds.append("") 

                else: 
                    batch_preds = decode_predictions(log_probs, int_to_char_inf, blank_index_inf)

                for test_id, pred in zip(ids_batch, batch_preds):
                    submission_results.append({'id': str(test_id), main_config.TARGET_COLUMN: pred})

            except Exception as e:
                print(f"\nОшибка инференса на батче {batch_idx}: {e}")
                for test_id in ids_batch:
                    submission_results.append({'id': str(test_id), main_config.TARGET_COLUMN: '[INFERENCE_ERROR]'})
                continue 

    inference_time = time.time() - inference_start_time
    print(f"Инференс завершен за {inference_time:.1f} сек. Предсказаний: {len(submission_results)}")

    if not submission_results:
        print("Нет результатов для создания submission файла.")
        return None

    submission_df = pd.DataFrame(submission_results)
    print(f"Получено {len(submission_df)} предсказаний.")

    if submission_df['id'].duplicated().any():
        print(f"Предупреждение: Найдены дубликаты ID ({submission_df['id'].duplicated().sum()} шт.). Удаляем дубликаты, оставляя первое вхождение.")
        submission_df = submission_df.drop_duplicates(subset=['id'], keep='first')

    final_submission_df = submission_df
    try:
        sample_df = pd.read_csv(main_config.SAMPLE_SUBMISSION_PATH)
        print(f"Загружен sample submission: {main_config.SAMPLE_SUBMISSION_PATH} ({len(sample_df)} записей)")
        submission_df['id'] = submission_df['id'].astype(str)
        sample_df['id'] = sample_df['id'].astype(str)

        final_submission_df = pd.merge(sample_df[['id']], submission_df[['id', main_config.TARGET_COLUMN]], on='id', how='left')

        missing_preds = final_submission_df[main_config.TARGET_COLUMN].isnull().sum()
        if missing_preds > 0:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {missing_preds} ID из sample_submission не найдены в предсказаниях. Заполнены пустыми строками.")
            final_submission_df[main_config.TARGET_COLUMN] = final_submission_df[main_config.TARGET_COLUMN].fillna('')

        if len(final_submission_df) != len(sample_df):
            print(f"ПРЕДУПРЕЖДЕНИЕ: Итоговый размер сабмита ({len(final_submission_df)}) не совпадает с sample_submission ({len(sample_df)})!")
        else:
            print("Размер итогового сабмита совпадает с sample_submission.")

    except FileNotFoundError:
        print(f"Не найден файл {main_config.SAMPLE_SUBMISSION_PATH}. Сабмит будет сохранен с ID, полученными во время инференса.")
        final_submission_df = submission_df[['id', main_config.TARGET_COLUMN]]
    except Exception as e:
        print(f"Ошибка при обработке sample_submission: {e}")
        final_submission_df = submission_df[['id', main_config.TARGET_COLUMN]]

    try:
        final_submission_df[['id', main_config.TARGET_COLUMN]].to_csv(main_config.SUBMISSION_FILE, index=False)
        print(f"\nФайл submission сохранен: {main_config.SUBMISSION_FILE}")
        print("Первые 5 строк:")
        print(final_submission_df[['id', main_config.TARGET_COLUMN]].head())
        return main_config.SUBMISSION_FILE
    except Exception as e:
        print(f"Ошибка сохранения submission файла: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":

    run_inference()