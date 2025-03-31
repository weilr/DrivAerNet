import logging
import os
import sys
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm


def init_logger(log_dir: str):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f'run_{timestamp}.log')

    # 设置 logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 清除已有 handler，避免重复打印
    if logger.hasHandlers():
        logger.handlers.clear()

    # 输出到文件
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    # 输出到控制台
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter('%(message)s'))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logging.info(f"[Logger] Initialization completed：{log_path}")


def log_tqdm(iterable, desc=None):
    return tqdm(iterable, desc=desc, file=sys.stdout, disable=not sys.stdout.isatty(), leave=False)


def progress(iterable, desc="", log_interval=10):
    """
    智能进度显示器：
    - 本地交互环境下使用 tqdm
    - 非交互环境下用 logging 输出每隔 log_interval 次进度
    """
    total = len(iterable) if hasattr(iterable, '__len__') else None
    if sys.stdout.isatty() or 'PYCHARM_HOSTED' in os.environ:
        yield from tqdm(iterable, desc=desc, file=sys.stdout, leave=False)
    else:
        logging.info(f"[progress] {desc}")
        for i, item in enumerate(iterable):
            if i % log_interval == 0 or i == total - 1:
                logging.info(f"[progress] {desc} [{i + 1}/{total}]")
            yield item


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model.state_dict(), path+'/'+'checkpoint.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss